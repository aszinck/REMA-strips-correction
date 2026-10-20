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
