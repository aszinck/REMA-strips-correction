import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

import preprocessing_tools as ptools

SOURCE_COLORS = {"SWATH": "#2a78d6", "POCA": "#eb6834"}
REJECTED_COLOR = "#8a8984"

#################################################
#######   Co-registration diagnostics    #######
#################################################

def read_dem_preview(path, max_pixels=800):
    """Read a decimated copy of a DEM (nearest neighbour, so blunders stay visible)."""
    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / max_pixels
        shape = (max(1, int(src.height / scale)), max(1, int(src.width / scale)))
        dem = src.read(1, out_shape=shape, masked=True)
        bounds = src.bounds
    dem = np.ma.masked_invalid(dem)
    xs = np.linspace(bounds.left, bounds.right, shape[1], endpoint=False) + (bounds.right - bounds.left) / shape[1] / 2
    ys = np.linspace(bounds.top, bounds.bottom, shape[0], endpoint=False) - (bounds.top - bounds.bottom) / shape[0] / 2
    return dem, xs, ys, bounds


def strip_axes(xx, yy, mask):
    """Along- and across-track unit vectors from the principal axes of the valid DEM pixels."""
    xy = np.column_stack((xx[~mask], yy[~mask]))
    centre = xy.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((xy - centre).T))
    along, across = vectors[:, 1], vectors[:, 0]
    return centre, along, across


def plot_strip_diagnostics(strip, qc_row, region_id, projectDir, fit_params=None, max_pixels=800):
    """Diagnostic panel for the co-registration of one strip.

    qc_row is the strip's row from the QC table, and fit_params the fit parameters
    used in the `coregistration` call (poca_total_weight, swath_total_weight,
    second_pass_threshold_m, min_points_for_plane).
    """
    fit_params = fit_params or {}
    points, fit = ptools.coreg_reference_points(strip, region_id, projectDir, **fit_params)
    path = ptools._coreg_paths(region_id, projectDir)["strip_dir"] / strip
    dem, xs, ys, bounds = read_dem_preview(path, max_pixels)
    xx, yy = np.meshgrid(xs, ys)
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    mask = np.ma.getmaskarray(dem)

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
    fig.suptitle(f"{strip}\nstatus: {qc_row['status']}   flags: {qc_row['quality_flags'] or '(none)'}",
                 x=0.01, ha="left", fontsize=11)

    # (a) DEM
    ax = axes[0, 0]
    vmin, vmax = np.nanpercentile(dem.compressed(), [2, 98]) if dem.count() else (0, 1)
    im = ax.imshow(dem, extent=extent, cmap="Greys_r", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Elevation (m)")
    ax.set_title("Tide-corrected DEM (2–98 % stretch)", loc="left")

    if fit is None:
        for ax in axes.flat[1:]:
            ax.axis("off")
        axes[0, 1].text(0, 0.5, "No reference observations", fontsize=12)
        _format_map_axes(axes[0, 0], _valid_extent(xx, yy, mask, extent))
        plt.tight_layout()
        return fig, points

    # (b, c) residuals before and after on the map, on a common colour scale
    limit = np.nanpercentile(np.abs(points["residual"]), 95)
    norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    order = np.argsort(np.abs(points["residual"].to_numpy()))
    for ax, col, title in [(axes[0, 1], "residual", "Residual before (REMA − reference)"),
                           (axes[0, 2], "residual_after", "Residual after")]:
        ax.imshow(dem, extent=extent, cmap="Greys_r", vmin=vmin, vmax=vmax, alpha=0.35)
        d = points.iloc[order]
        sc = ax.scatter(d["x"], d["y"], c=d[col], cmap="RdBu_r", norm=norm, s=6, linewidth=0)
        fig.colorbar(sc, ax=ax, shrink=0.8, extend="both", label="Residual (m)")
        ax.set_title(title, loc="left")

    # (d) correction plane over the valid footprint
    ax = axes[0, 3]
    plane = fit["a"] * (xx - fit["x0"]) + fit["b"] * (yy - fit["y0"]) + fit["c"]
    plane = np.ma.array(plane, mask=mask)
    plane_limit = max(np.abs(plane).max(), 1e-3)
    im = ax.imshow(plane, extent=extent, cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-plane_limit, vmax=plane_limit))
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correction subtracted (m)")
    ax.set_title(f"Correction plane ({fit['fit_mode']})", loc="left")

    for ax in axes[0]:
        _format_map_axes(ax, _valid_extent(xx, yy, mask, extent))

    # (e, f) residuals along and across the strip
    centre, along, across = strip_axes(xx, yy, mask)
    rel = np.column_stack((points["x"], points["y"])) - centre
    points = points.assign(along_km=rel @ along / 1000, across_km=rel @ across / 1000)
    kept = points["kept"].to_numpy()
    for ax, col, label in [(axes[1, 0], "along_km", "Along-track distance (km)"),
                           (axes[1, 1], "across_km", "Across-track distance (km)")]:
        ax.scatter(points[col], points["residual"], s=5, color="#d6d5d0", linewidth=0)
        ax.scatter(points.loc[~kept, col], points.loc[~kept, "residual_after"], s=10,
                   marker="x", color=REJECTED_COLOR, linewidth=0.7)
        for source, color in SOURCE_COLORS.items():
            d = points[kept & (points["source"] == source).to_numpy()]
            ax.scatter(d[col], d["residual_after"], s=6, color=color, linewidth=0)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set(xlabel=label, ylabel="Residual (m)", ylim=(-2 * limit, 2 * limit))
    axes[1, 0].set_title("Residuals along the strip", loc="left")
    axes[1, 1].set_title("Residuals across the strip", loc="left")
    axes[1, 0].legend(handles=[
        Line2D([], [], linestyle="", marker="o", markersize=5, color="#d6d5d0", label="before"),
        *[Line2D([], [], linestyle="", marker="o", markersize=5, color=c, label=f"{s} after")
          for s, c in SOURCE_COLORS.items()],
        Line2D([], [], linestyle="", marker="x", markersize=6, color=REJECTED_COLOR, label="rejected, after"),
    ], frameon=False, loc="upper left", fontsize=8)

    # (g) residual distributions
    ax = axes[1, 2]
    bins = np.linspace(-2 * limit, 2 * limit, 60)
    ax.hist(points["residual"], bins=bins, histtype="step", color=REJECTED_COLOR, linewidth=1.5, label="before")
    for source, color in SOURCE_COLORS.items():
        d = points[kept & (points["source"] == source).to_numpy()]
        if len(d):
            ax.hist(d["residual_after"], bins=bins, histtype="step", color=color, linewidth=1.5,
                    label=f"{source} after (kept)")
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set(xlabel="Residual (m)", ylabel="Number of points")
    ax.set_title("Residual distributions", loc="left")
    ax.legend(frameon=False, fontsize=8)

    # (h) summary
    ax = axes[1, 3]
    ax.axis("off")
    n_kept = int(kept.sum())
    lines = [
        f"Acquisition: {str(qc_row['acquisition_time_utc'])[:10]}",
        f"Reference months: {qc_row['months']}",
        f"Points: {int(qc_row['n_poca'])} POCA, {int(qc_row['n_swath'])} SWATH",
        f"Kept in 2nd pass: {n_kept} of {len(points)} ({n_kept / len(points):.0%})",
        f"Coverage: {qc_row['combined_coverage_fraction']:.0%} of footprint, "
        f"{int(qc_row['n_covered_quadrants'])}/4 quadrants",
        "",
        f"RMSE before: {qc_row['rmse_before']:.1f} m",
        f"RMSE after:  {qc_row['rmse_after']:.1f} m",
        f"  POCA:  {qc_row['rmse_poca_before']:.1f} → {qc_row['rmse_poca_after']:.1f} m",
        f"  SWATH: {qc_row['rmse_swath_before']:.1f} → {qc_row['rmse_swath_after']:.1f} m",
        "",
        f"Offset c: {fit['c']:.1f} m",
        f"Tilt: {1000 * np.hypot(fit['a'], fit['b']):.2f} m/km",
        f"Correction on footprint: {plane.min():.1f} to {plane.max():.1f} m",
    ]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)

    for ax in axes[1, :3]:
        ax.grid(True, color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)

    plt.tight_layout()
    return fig, points


def _valid_extent(xx, yy, mask, extent, pad=0.03):
    if mask.all():
        return extent
    x, y = xx[~mask], yy[~mask]
    dx, dy = pad * (x.max() - x.min()), pad * (y.max() - y.min())
    return (x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy)


def _format_map_axes(ax, extent):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    km = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(km)
        axis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("x (km, EPSG:3031)", fontsize=8)
    ax.set_ylabel("y (km, EPSG:3031)", fontsize=8)


#################################################
#######     Monthly movie of the DEMs    #######
#################################################

def _reproject_to(path, dst_transform, shape, resampling, nodata=-9999.):
    from rasterio.warp import reproject
    out = np.full(shape, np.nan, dtype='float32')
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, src_nodata=src.nodata if src.nodata is not None else nodata,
                  dst_transform=dst_transform, dst_crs='EPSG:3031', dst_nodata=np.nan, resampling=resampling)
    return out


TIER_B_COLOR = "#eb6834"
TIER_C_COLOR = "#e87ba4"
# Outline styles in the movie; TIER_A strips are not outlined
OUTLINE_STYLE = {'TIER_B': dict(colors=TIER_B_COLOR, linestyles='--'),
                 'TIER_C': dict(colors=TIER_C_COLOR, linestyles=':')}


def _footprint(grid):
    """Outline-friendly footprint of a strip: valid pixels with small gaps closed and holes filled."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    return binary_fill_holes(binary_closing(np.isfinite(grid), iterations=4))


def _adjusted_strips(region_id, projectDir, tiers):
    import pandas as pd
    from pathlib import Path
    adj_dir = Path(projectDir) / 'data/interim' / f'region-{region_id}' / 'REMA/adjusted-strips'
    table = pd.read_csv(adj_dir / 'adjustment_strips.csv')
    table = table[table['tier'].isin(tiers)].copy()
    table['time'] = pd.to_datetime(table['acquisition_time_utc'], utc=True)
    table['month'] = table['time'].dt.strftime('%Y-%m')
    return table.sort_values('time'), adj_dir


def _frame_grid(region_id, projectDir, buffer_m, resolution_m):
    """The AOI polygon and a grid covering it plus buffer_m."""
    import geopandas as gpd
    from pathlib import Path
    from rasterio.transform import from_origin
    polygon = gpd.read_file(Path(projectDir) / 'data/initial/regions' / f'region-{region_id}.shp').to_crs('EPSG:3031')
    left, bottom, right, top = polygon.buffer(buffer_m).total_bounds
    left, top = np.floor(left / resolution_m) * resolution_m, np.ceil(top / resolution_m) * resolution_m
    width = int(np.ceil((right - left) / resolution_m))
    height = int(np.ceil((top - bottom) / resolution_m))
    extent = (left, left + width * resolution_m, top - height * resolution_m, top)
    return polygon, from_origin(left, top, resolution_m, resolution_m), (height, width), extent


def _colour_range(grids, polygon, transform, shape):
    """2nd-98th percentile of all strips inside the polygon."""
    from rasterio.features import geometry_mask
    inside = ~geometry_mask(polygon.geometry, shape, transform)
    values = np.concatenate([g[inside & np.isfinite(g)] for g in grids])
    return tuple(np.percentile(values, [2, 98]))


def movie_colour_range(region_id, projectDir, tiers=("TIER_A", "TIER_B", "TIER_C"), buffer_m=10000,
                       resolution_m=50):
    """The default colour range of dem_movie, to use the same range elsewhere."""
    from rasterio.enums import Resampling
    table, _ = _adjusted_strips(region_id, projectDir, tiers)
    polygon, transform, shape, _ = _frame_grid(region_id, projectDir, buffer_m, resolution_m)
    grids = [_reproject_to(f, transform, shape, Resampling.average) for f in table['output_file']]
    return _colour_range(grids, polygon, transform, shape)


def dem_movie(region_id, projectDir, mosaic_file, output_file=None, tiers=("TIER_A", "TIER_B", "TIER_C"),
              buffer_m=10000, resolution_m=50, fps=2, vmin=None, vmax=None,
              include_empty_months=False, dpi=150):
    """MP4 movie of the adjusted strips, one frame per month, over a greyscale REMA mosaic.

    Strips from the same month are drawn together, later acquisitions on top. The colour
    range is the same in all frames (by default the 2nd-98th percentile of all strips
    inside the AOI polygon). TIER_B strips are outlined with a dashed orange line, and
    TIER_C (cross-track) strips with a dotted pink line.
    """
    import pandas as pd
    from pathlib import Path
    from rasterio.enums import Resampling
    from matplotlib.animation import FFMpegWriter

    table, adj_dir = _adjusted_strips(region_id, projectDir, tiers)
    output_file = Path(output_file) if output_file else adj_dir / f'adjusted_strips_monthly_{region_id}.mp4'
    polygon, transform, (height, width), extent = _frame_grid(region_id, projectDir, buffer_m, resolution_m)
    left, top = extent[0], extent[3]

    print(f'Reading {len(table)} strips at {resolution_m} m')
    grids = {row.strip: _reproject_to(row.output_file, transform, (height, width), Resampling.average)
             for row in table.itertuples()}
    background = _reproject_to(mosaic_file, transform, (height, width), Resampling.bilinear)
    if vmin is None or vmax is None:
        low, high = _colour_range(grids.values(), polygon, transform, (height, width))
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
    bg_low, bg_high = np.nanpercentile(background, [2, 98])

    months = sorted(table['month'].unique())
    if include_empty_months:
        months = pd.period_range(months[0], months[-1], freq='M').strftime('%Y-%m').tolist()

    # Map on top, colour bar to the right, list of strips underneath
    map_width = 7.5
    map_height = map_width * height / width
    fw, fh = map_width + 2.2, map_height + 2.6
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0.9 / fw, 1.9 / fh, map_width / fw, map_height / fh])
    cax = fig.add_axes([(map_width + 1.1) / fw, 1.9 / fh, 0.2 / fw, map_height / fh])
    ax.imshow(background, extent=extent, cmap='gray', vmin=bg_low, vmax=bg_high)
    image = ax.imshow(np.full((height, width), np.nan), extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
    polygon.boundary.plot(ax=ax, color='white', linewidth=1.5)
    ax.legend(handles=[Line2D([], [], color='white', linewidth=1.5, label='AOI polygon'),
                       Line2D([], [], color=TIER_B_COLOR, linewidth=1.2, linestyle='--', label='TIER_B strip'),
                       Line2D([], [], color=TIER_C_COLOR, linewidth=1.5, linestyle=':', label='TIER_C strip')],
              loc='lower left', bbox_to_anchor=(0, 1.005), ncol=3, frameon=True, facecolor='#8a8984',
              edgecolor='none', labelcolor='white', fontsize=8, borderaxespad=0)
    fig.colorbar(image, cax=cax, label='Elevation (m, WGS84 ellipsoid)')
    _format_map_axes(ax, extent)
    ax.set_xlabel('x (km, EPSG:3031)')
    ax.set_ylabel('y (km, EPSG:3031)')
    title = ax.set_title('', loc='right', fontsize=14)
    caption = fig.text(0.9 / fw, 1.45 / fh, '', fontsize=7, va='top', family='monospace')
    xs = left + (np.arange(width) + .5) * resolution_m
    ys = top - (np.arange(height) + .5) * resolution_m
    outlines = []

    output_file.parent.mkdir(parents=True, exist_ok=True)
    # H.264 needs an even number of pixels in both directions, so pad by one pixel if needed
    writer = FFMpegWriter(fps=fps, codec='libx264',
                          extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white', '-pix_fmt', 'yuv420p'])
    with writer.saving(fig, str(output_file), dpi):
        for month in months:
            strips = table[table['month'] == month]
            composite = np.full((height, width), np.nan, dtype='float32')
            for row in strips.itertuples():
                g = grids[row.strip]
                composite = np.where(np.isfinite(g), g, composite)
            image.set_data(composite)
            for o in outlines:
                o.remove()
            outlines = [ax.contour(xs, ys, _footprint(grids[row.strip]).astype(float), levels=[0.5],
                                   linewidths=1.2 if row.tier == 'TIER_B' else 1.5, **OUTLINE_STYLE[row.tier])
                        for row in strips.itertuples() if row.tier in OUTLINE_STYLE]
            title.set_text(pd.Period(month).strftime('%B %Y'))
            caption.set_text('\n'.join(f"{row.time:%Y-%m-%d}  {row.tier}  {row.strip.split('_dem')[0]}"
                                       for row in strips.itertuples()) or 'no strips')
            writer.grab_frame()
    plt.close(fig)
    print(f'Wrote {len(months)} frames to {output_file}')
    return output_file


def plot_month_strips(region_id, projectDir, month, mosaic_file, vmin=None, vmax=None,
                      buffer_m=10000, resolution_m=25, ncols=2, output_file=None, dpi=110,
                      tiers=("TIER_A", "TIER_B", "TIER_C", "DISCARDED"), strips=None):
    """All strips from one month (e.g. '2010-11'), each in its own panel with its QC stats.

    Uses the adjusted strip where there is one, and the tide-corrected strip for discarded
    strips. Drawn with nearest-neighbour resampling so blunders stay visible. The colour
    range defaults to that of dem_movie. Use tiers or strips (file names) to plot only some.
    """
    import pandas as pd
    from rasterio.enums import Resampling

    table, adj_dir = _adjusted_strips(region_id, projectDir, tiers)
    table = table[table['month'] == month]
    if strips is not None:
        table = table[table['strip'].isin(strips)]
    if table.empty:
        raise ValueError(f'No strips in {month}')
    coreg = pd.read_csv(ptools._coreg_paths(region_id, projectDir)['output_dir'] / 'coreg_quality_control.csv')
    coreg = coreg.set_index('strip')
    strip_dir = ptools._coreg_paths(region_id, projectDir)['strip_dir']
    if vmin is None or vmax is None:
        low, high = movie_colour_range(region_id, projectDir, buffer_m=buffer_m)
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax

    polygon, transform, shape, extent = _frame_grid(region_id, projectDir, buffer_m, resolution_m)
    background = _reproject_to(mosaic_file, transform, shape, Resampling.bilinear)
    bg_low, bg_high = np.nanpercentile(background, [2, 98])

    # One block per strip: map (zoomed to the strip), colour bar, stats. Sizes in inches.
    map_w, cbar_w, text_w, gap = 6.0, 0.15, 4.6, 0.9
    map_h, title_h = 5.0, 0.5
    block_w, block_h = map_w + cbar_w + text_w + 2 * gap, map_h + title_h + 0.6
    ncols = min(ncols, len(table))
    nrows = int(np.ceil(len(table) / ncols))
    fw, fh = ncols * block_w + 0.3, nrows * block_h + 0.5
    fig = plt.figure(figsize=(fw, fh))
    xs = extent[0] + (np.arange(shape[1]) + .5) * resolution_m
    ys = extent[3] - (np.arange(shape[0]) + .5) * resolution_m
    for n, row in enumerate(table.itertuples()):
        x0 = 0.3 + (n % ncols) * block_w + 0.6
        y0 = fh - 0.5 - (n // ncols + 1) * block_h + 0.6
        ax = fig.add_axes([x0 / fw, y0 / fh, map_w / fw, map_h / fh])
        cax = fig.add_axes([(x0 + map_w + 0.15) / fw, y0 / fh, cbar_w / fw, map_h / fh])
        text_x = (x0 + map_w + cbar_w + gap) / fw
        adjusted = isinstance(row.output_file, str)
        path = row.output_file if adjusted else strip_dir / row.strip
        dem = _reproject_to(path, transform, shape, Resampling.nearest)
        ax.imshow(background, extent=extent, cmap='gray', vmin=bg_low, vmax=bg_high)
        image = ax.imshow(dem, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
        polygon.boundary.plot(ax=ax, color='white', linewidth=1.2)
        # Zoom to the strip, keeping the map's aspect ratio
        valid = np.isfinite(dem)
        if valid.any():
            cols, rows = np.flatnonzero(valid.any(axis=0)), np.flatnonzero(valid.any(axis=1))
            cx, cy = (xs[cols[0]] + xs[cols[-1]]) / 2, (ys[rows[0]] + ys[rows[-1]]) / 2
            half_w = max(xs[cols[-1]] - xs[cols[0]], (ys[rows[0]] - ys[rows[-1]]) * map_w / map_h) / 2 + 1000
            half_h = half_w * map_h / map_w
            zoom = (cx - half_w, cx + half_w, cy - half_h, cy + half_h)
        else:
            zoom = extent
        _format_map_axes(ax, zoom)
        ax.set_aspect('auto')
        colour = {'TIER_A': '#0ca30c', 'TIER_B': '#eda100', 'TIER_C': '#d55181', 'DISCARDED': '#d03b3b'}[row.tier]
        ax.set_title(f"{row.time:%Y-%m-%d %H:%M}  {row.tier}" + ('' if adjusted else '  (not adjusted)'),
                     loc='left', fontsize=11, color=colour, fontweight='bold')
        fig.colorbar(image, cax=cax, label='Elevation (m)')

        c = coreg.loc[row.strip] if row.strip in coreg.index else None
        fmt = lambda v, f='.2f': '–' if pd.isna(v) else format(v, f)
        lines = [
            row.strip.split('_dem')[0].replace('SETSM_s2s041_', ''),
            '',
            f"tier:           {row.tier}",
            f"discard reason: {row.discard_reason if isinstance(row.discard_reason, str) else '–'}",
            f"tier B reason:  {row.tier_b_reason if isinstance(row.tier_b_reason, str) else '–'}",
            f"cross-track:    {'yes' if row.is_xtrack else 'no'}",
            '',
            'Inside the polygon (step 1)',
            f"  coverage:          {fmt(row.polygon_coverage)}",
            f"  CryoSat-2 points:  {int(row.n_ref_polygon)}",
            f"  NMAD:              {fmt(row.ref_nmad_polygon)} m",
            f"  outlier fraction:  {fmt(row.ref_outlier_fraction_polygon)}",
            f"  pixel noise:       {fmt(row.noise_nmad_m)} m",
            '',
            'Adjustment (steps 2-3)',
            f"  pairs:             {int(row.n_pairs)}",
            f"  overlap outliers:  {fmt(row.overlap_outlier_fraction)}",
            f"  max pair offset:   {fmt(row.max_pair_offset_after_m)} m",
            f"  CryoSat-2 used:    {int(row.n_ref_used)}",
            f"  NMAD before/after: {fmt(row.ref_nmad_before_m)} / {fmt(row.ref_nmad_after_m)} m",
            f"  sigma in polygon:  {fmt(row.sigma_polygon_max_m)} m",
            f"  offset c:          {fmt(row.c_m, '.1f')} m",
            f"  tilt a, b:         {fmt(row.a_m_per_km)}, {fmt(row.b_m_per_km)} m/km",
            '',
            'Per-strip co-registration',
            f"  status: {c['status'] if c is not None else '–'}",
            f"  flags:  {(c['quality_flags'] if c is not None and isinstance(c['quality_flags'], str) else '–').replace(';', ', ')}",
        ]
        fig.text(text_x, (y0 + map_h) / fh, '\n'.join(lines), va='top', family='monospace', fontsize=8)
    heading = (f'All strips from {pd.Period(month).strftime("%B %Y")}' if strips is None and len(tiers) == 4
               else f'Strips from {pd.Period(month).strftime("%B %Y")}')
    fig.suptitle(f'{heading} (colour range {vmin:.0f} to {vmax:.0f} m, each map zoomed to its strip)',
                 x=0.3 / fw, y=1 - 0.25 / fh, ha='left', fontsize=13)
    if output_file:
        fig.savefig(output_file, dpi=dpi)
    return fig
