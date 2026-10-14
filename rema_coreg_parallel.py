"""Memory-aware REMA/CryoSat-2 co-registration for Pine Island Glacier.

Requires numpy, pandas, geopandas, xarray, rasterio, pyarrow.
Input REMA strips are assumed tide-corrected, as are POCA/swath heights.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import xarray as xr

# ========================= CONFIGURATION =========================

PROJECT_DIR = Path('.')
REGION_ID = 'PIG'
REMA_ROOT = PROJECT_DIR / 'data/initial/REMA' / f'region-{REGION_ID}'
STRIP_DIR = REMA_ROOT / 'REMA_strips_tide_corrected'
METADATA_DIR = REMA_ROOT / 'metadataFiles'
POCA_DIR = PROJECT_DIR / 'data/L2_poca_corrected'
SWATH_FILE = PROJECT_DIR / 'data/L2_swath_corrected/swath_corrected_monthly_500m.nc'
OUTPUT_DIR = PROJECT_DIR / 'data/interim' / f'region-{REGION_ID}' / 'REMA/co-registered-strips'

N_WORKERS = 3
QC_RESOLUTION_M = 2000
POCA_TOTAL_WEIGHT = 5.0
SWATH_TOTAL_WEIGHT = 1.0
SECOND_PASS_THRESHOLD_M = 15.0
MIN_POINTS_FOR_PLANE = 10
MIN_POCA = 10
MIN_SWATH = 20
MIN_FOOTPRINT_COVERAGE = 0.20
MIN_COVERED_QUADRANTS = 3
MAX_RMSE_M = 10.0
OVERWRITE = False

# Avoid BLAS thread oversubscription when running multiple processes.
for _var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_var, '1')


def base_name(filename):
    return re.sub(r'_dem(?:_[0-9.]+m)?\.tif$', '', Path(filename).name)


def acquisition_midpoint(path):
    text = Path(path).read_text()
    times = []
    for key in ('avgAcqTime1', 'avgAcqTime2'):
        m = re.search(rf'{key}\s*=\s*(\d{{4}}-\d{{2}}-\d{{2}}[ T]\d{{2}}:\d{{2}}:\d{{2}}(?:\.\d+)?)', text)
        if not m:
            raise ValueError(f'Missing {key} in {path}')
        times.append(pd.Timestamp(m.group(1), tz='UTC'))
    return times[0] + (times[1] - times[0]) / 2


def months_for_date(date):
    p = pd.Timestamp(date).tz_localize(None).to_period('M')
    periods = (p - 1, p) if date.day <= 15 else (p, p + 1)
    return [(m.year, m.month) for m in periods]


def _empty(source):
    return pd.DataFrame({'x': pd.Series(dtype='float64'),
                         'y': pd.Series(dtype='float64'),
                         'h': pd.Series(dtype='float64'),
                         'source': pd.Series(dtype='object')})


def load_poca(months, bounds):
    frames = []
    for year, month in months:
        path = POCA_DIR / str(year) / f'{month:02d}' / f'poca_corrected_{year}_{month:02d}.feather'
        if not path.exists():
            continue
        gdf = gpd.read_feather(path)
        if gdf.crs is None:
            raise ValueError(f'POCA CRS missing: {path}')
        if gdf.crs.to_epsg() != 3031:
            gdf = gdf.to_crs('EPSG:3031')
        gdf = gdf.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
        if not gdf.empty:
            frames.append(pd.DataFrame({'x': gdf.geometry.x.to_numpy(),
                                        'y': gdf.geometry.y.to_numpy(),
                                        'h': gdf.h_corrected.to_numpy(),
                                        'source': 'POCA'}))
    if not frames:
        return _empty('POCA')
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna(subset=['x','y','h'])


def load_swath_median(months, bounds):
    """Median across the selected two monthly grids; one record per cell."""
    dates = [np.datetime64(f'{y:04d}-{m:02d}-15') for y, m in months]
    with xr.open_dataset(SWATH_FILE) as ds:
        da = ds['h_corrected']
        available = da.time.values.astype('datetime64[ns]')
        selected = [d for d in dates if np.isin(d.astype('datetime64[ns]'), available)]
        if not selected:
            return _empty('SWATH')
        xs = slice(bounds.left, bounds.right) if da.x.values[0] < da.x.values[-1] else slice(bounds.right, bounds.left)
        ys = slice(bounds.bottom, bounds.top) if da.y.values[0] < da.y.values[-1] else slice(bounds.top, bounds.bottom)
        subset = da.sel(time=selected, x=xs, y=ys)
        if not subset.sizes.get('x', 0) or not subset.sizes.get('y', 0):
            return _empty('SWATH')
        # NumPy median is faster and avoids building a large lazy xarray graph.
        stack = subset.to_numpy()
        with np.errstate(all='ignore'):
            if stack.shape[0] == 1:
                median = stack[0]
            else:
                a, b = stack[0], stack[1]
                median = np.where(np.isfinite(a) & np.isfinite(b), (a + b) / 2,
                                  np.where(np.isfinite(a), a, b))
        yy, xx = np.meshgrid(subset.y.values, subset.x.values, indexing='ij')
        good = np.isfinite(median)
        return pd.DataFrame({'x': xx[good], 'y': yy[good], 'h': median[good], 'source': 'SWATH'})


def sample_dem(path, data):
    if data.empty:
        return data.copy()
    data = data.copy()
    with rasterio.open(path) as src:
        coords = zip(data.x.to_numpy(), data.y.to_numpy())
        sampled = np.ma.array([v[0] for v in src.sample(coords, masked=True)], dtype='float64')
        values = np.asarray(sampled.filled(np.nan), dtype='float64')
        if src.nodata is not None and np.isfinite(src.nodata):
            values[values == src.nodata] = np.nan
    data['h_rema'] = values
    data['residual'] = values - data.h.to_numpy()
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=['residual'])


def source_weights(data):
    src = data.source.to_numpy()
    w = np.zeros(len(data), dtype='float64')
    for name, total in [('POCA', POCA_TOTAL_WEIGHT), ('SWATH', SWATH_TOTAL_WEIGHT)]:
        mask = src == name
        if mask.any():
            w[mask] = total / mask.sum()
    return w


def fit(data, x0, y0, offset_only=False):
    x = data.x.to_numpy() - x0
    y = data.y.to_numpy() - y0
    z = data.residual.to_numpy()
    w = source_weights(data)
    if offset_only:
        return np.array([0., 0., np.average(z, weights=w)])
    # Scale coordinates to kilometres for improved conditioning; convert slopes back.
    X = np.column_stack((x / 1000., y / 1000., np.ones(len(data))))
    A = X * np.sqrt(w)[:, None]
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError('Rank-deficient spatial coverage')
    coefs = np.linalg.lstsq(A, z * np.sqrt(w), rcond=None)[0]
    return np.array([coefs[0] / 1000., coefs[1] / 1000., coefs[2]])


def corrected_residual(data, coeff, x0, y0):
    a, b, c = coeff
    return data.residual.to_numpy() - (a * (data.x.to_numpy() - x0) + b * (data.y.to_numpy() - y0) + c)


def rmse(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2))) if len(values) else np.nan


def weighted_rmse(values, weights):
    return float(np.sqrt(np.average(np.asarray(values) ** 2, weights=weights))) if len(values) else np.nan


def fit_two_pass(data, bounds):
    x0 = (bounds.left + bounds.right) / 2
    y0 = (bounds.bottom + bounds.top) / 2
    mode = 'plane'
    if len(data) < MIN_POINTS_FOR_PLANE:
        mode = 'offset_only'
    try:
        first = fit(data, x0, y0, offset_only=(mode == 'offset_only'))
    except ValueError:
        mode = 'offset_only'
        first = fit(data, x0, y0, offset_only=True)
    first_resid = corrected_residual(data, first, x0, y0)
    inliers = np.abs(first_resid) <= SECOND_PASS_THRESHOLD_M
    subset = data.iloc[np.flatnonzero(inliers)]
    final = first
    if len(subset) >= (MIN_POINTS_FOR_PLANE if mode == 'plane' else 1):
        try:
            final = fit(subset, x0, y0, offset_only=(mode == 'offset_only'))
        except ValueError:
            mode = 'offset_only'
            final = fit(subset, x0, y0, offset_only=True)
    return final, x0, y0, mode, inliers


def footprint_coverage(path, poca, swath):
    """2 km footprint grid, weighted by actual valid DEM fraction in each cell.

    Rasterio average-resamples the DEM valid-data mask (0..255), so partial
    footprint cells contribute proportionally instead of as full 2 km cells.
    """
    with rasterio.open(path) as src:
        dx = abs(src.transform.a)
        dy = abs(src.transform.e)
        out_width = max(1, int(np.ceil(src.width * dx / QC_RESOLUTION_M)))
        out_height = max(1, int(np.ceil(src.height * dy / QC_RESOLUTION_M)))
        valid_fraction = src.read_masks(1, out_shape=(out_height, out_width),
                                        resampling=Resampling.average).astype('float32') / 255.0
        from rasterio.transform import from_bounds, rowcol
        qc_transform = from_bounds(*src.bounds, out_width, out_height)
        bounds = src.bounds

    footprint = valid_fraction > 0
    total_area = float(valid_fraction.sum())
    out = {'valid_footprint_km2': total_area * abs(bounds.right-bounds.left) * abs(bounds.top-bounds.bottom) / (out_width*out_height) / 1e6,
           'n_valid_qc_cells': int(footprint.sum())}

    def occupied(data):
        occupancy = np.zeros((out_height, out_width), dtype=bool)
        if data.empty:
            return occupancy
        rr, cc = rowcol(qc_transform, data.x.to_numpy(), data.y.to_numpy())
        rr, cc = np.asarray(rr), np.asarray(cc)
        good = (rr >= 0) & (rr < out_height) & (cc >= 0) & (cc < out_width)
        occupancy[rr[good], cc[good]] = True
        return occupancy & footprint

    p = occupied(poca)
    s = occupied(swath)
    both = p | s
    for name, occ in [('poca', p), ('swath', s), ('combined', both)]:
        out[f'{name}_coverage_fraction'] = float(valid_fraction[occ].sum() / total_area) if total_area else 0.0
        out[f'{name}_covered_qc_cells'] = int(occ.sum())

    # Quadrants defined by the centre of the valid footprint, not the raster bounds.
    iy, ix = np.indices(valid_fraction.shape)
    if total_area:
        cx = float((ix * valid_fraction).sum() / total_area)
        cy = float((iy * valid_fraction).sum() / total_area)
    else:
        cx, cy = (out_width - 1) / 2, (out_height - 1) / 2
    quadrants = (ix >= cx).astype('uint8') + 2 * (iy >= cy).astype('uint8')
    out['n_covered_quadrants'] = 0
    for q in range(4):
        in_quad = (quadrants == q) & footprint
        area = float(valid_fraction[in_quad].sum())
        frac = float(valid_fraction[in_quad & both].sum() / area) if area else np.nan
        out[f'quadrant_{q+1}_coverage_fraction'] = frac
        if area and np.any(in_quad & both):
            out['n_covered_quadrants'] += 1
    return out


def write_corrected(input_path, output_path, coeff, x0, y0):
    """Windowed read/correct/write, never loading the full strip into RAM."""
    a, b, c = coeff
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + '.partial.tif')
    try:
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            profile.update(driver='GTiff', dtype='float32', compress='deflate',
                           predictor=3, tiled=True, blockxsize=512, blockysize=512,
                           nodata=-9999.0, BIGTIFF='IF_SAFER')
            with rasterio.open(temp_path, 'w', **profile) as dst:
                for _, window in dst.block_windows(1):
                    raw = src.read(1, window=window, masked=True)
                    rows = np.arange(window.row_off, window.row_off + window.height)
                    cols = np.arange(window.col_off, window.col_off + window.width)
                    x = src.transform.c + (cols + .5) * src.transform.a
                    y = src.transform.f + (rows + .5) * src.transform.e
                    plane = a * (x[None, :] - x0) + b * (y[:, None] - y0) + c
                    corrected = np.asarray(raw, dtype='float64') - plane
                    corrected = np.where(np.ma.getmaskarray(raw) | ~np.isfinite(corrected), -9999., corrected)
                    dst.write(corrected.astype('float32'), 1, window=window)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def process_one(path_str):
    path = Path(path_str)
    row = {'strip': path.name, 'status': 'FAILED', 'quality_flags': '', 'error': ''}
    try:
        metadata = METADATA_DIR / f'{base_name(path.name)}_mdf.txt'
        when = acquisition_midpoint(metadata)
        months = months_for_date(when)
        row['acquisition_time_utc'] = when.isoformat()
        row['months'] = ';'.join(f'{y}-{m:02d}' for y, m in months)
        with rasterio.open(path) as src:
            if src.crs is None or src.crs.to_epsg() != 3031:
                raise ValueError('Expected REMA strip in EPSG:3031')
            bounds = src.bounds
        poca = sample_dem(path, load_poca(months, bounds))
        swath = sample_dem(path, load_swath_median(months, bounds))
        data = pd.concat([poca, swath], ignore_index=True)
        row.update(n_poca=len(poca), n_swath=len(swath), n_total=len(data))
        row.update(footprint_coverage(path, poca, swath))
        flags = []
        if len(poca) < MIN_POCA: flags.append('LOW_POCA_COUNT')
        if len(swath) < MIN_SWATH: flags.append('LOW_SWATH_COUNT')
        if row['combined_coverage_fraction'] < MIN_FOOTPRINT_COVERAGE: flags.append('LOW_FOOTPRINT_COVERAGE')
        if row['n_covered_quadrants'] < MIN_COVERED_QUADRANTS: flags.append('POOR_SPATIAL_DISTRIBUTION')
        if data.empty:
            flags.append('NO_REFERENCE_OBSERVATIONS')
            row['quality_flags'] = ';'.join(flags)
            row['status'] = 'UNCORRECTED'
            return row
        coeff, x0, y0, mode, inliers = fit_two_pass(data, bounds)
        if mode == 'offset_only': flags.append('OFFSET_ONLY')
        before = data.residual.to_numpy()
        after = corrected_residual(data, coeff, x0, y0)
        weights = source_weights(data)
        row.update(fit_mode=mode, a=float(coeff[0]), b=float(coeff[1]), c=float(coeff[2]),
                   x0=float(x0), y0=float(y0), n_second_pass=int(inliers.sum()),
                   n_rejected=int((~inliers).sum()),
                   rmse_before=rmse(before), rmse_after=rmse(after),
                   weighted_rmse_before=weighted_rmse(before, weights),
                   weighted_rmse_after=weighted_rmse(after, weights))
        for name in ('POCA', 'SWATH'):
            mask = data.source.to_numpy() == name
            row[f'rmse_{name.lower()}_before'] = rmse(before[mask])
            row[f'rmse_{name.lower()}_after'] = rmse(after[mask])
        if row['rmse_after'] > MAX_RMSE_M: flags.append('HIGH_RMSE')
        if row['rmse_after'] > row['rmse_before']: flags.append('RMSE_INCREASED')
        row['quality_flags'] = ';'.join(flags)
        output = OUTPUT_DIR / f'{path.stem}_coreg.tif'
        if OVERWRITE or not output.exists():
            write_corrected(path, output, coeff, x0, y0)
        row['output_file'] = str(output)
        row['status'] = 'FLAGGED' if flags else 'OK'
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        row['quality_flags'] = ';'.join(filter(None, [row.get('quality_flags'), 'PROCESSING_FAILED']))
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(STRIP_DIR.glob('*.tif'))
    if not paths:
        raise FileNotFoundError(f'No REMA strips found in {STRIP_DIR}')
    print(f'Processing {len(paths)} strips using {N_WORKERS} processes')
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_one, str(path)): path for path in paths}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f'[{i}/{len(paths)}] {row["strip"]}: {row["status"]} '
                  f'RMSE {row.get("rmse_before", np.nan):.2f} -> {row.get("rmse_after", np.nan):.2f} m '
                  f'({row.get("quality_flags", "")})', flush=True)
            # Write in main process only, preserving all completed QC results.
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'coreg_quality_control.csv', index=False)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
