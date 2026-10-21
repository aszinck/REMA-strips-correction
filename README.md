# Code to download and process REMA strips over ice shelves

This project is a preprocessing pipeline for **REMA strip DEMs** (Reference Elevation Model of Antarctica) over **ice shelves**. It downloads the strips for an area of interest, cleans them, removes the ocean tide signal, and co-registers them to CryoSat-2 altimetry so the elevations are consistent and comparable between strips.

The full workflow is in `data-preprocessing.ipynb`. Most of the logic lives in `preprocessing_tools.py`.

## Workflow

1. **Define an area of interest (AOI).** Give a lon/lat polygon and a region name (e.g. `"PIG"` for Pine Island Glacier). `polygon_coords_to_gdf` reprojects it to the project CRS (EPSG:3031 by default) and saves it.

2. **Download the strips.** `download-REMA-strips.py` finds the strips that overlap the AOI using the REMA strip index shapefile. It downloads them in parallel and can optionally resample them with GDAL. First download the REMA Strip DEM extent index shapefile from https://www.pgc.umn.edu/data/rema/ and save it as `{projectDir}/data/initial/REMA/REMA_Strip_Index_s2s041.shp`. Then run:

   ```
   python download-REMA-strips.py --region_id PIG --projectDir "/path/to/project" --resample true --resolution 50
   ```

   Flags:
   - `--region_id`: name of the region/AOI (required)
   - `--projectDir`: path to the project directory (required)
   - `--project_crs`: CRS of the project (default `EPSG:3031`)
   - `--n_threads`: number of parallel download threads (default 8)
   - `--resample`: whether to resample the strips, `true`/`false` (default `true`)
   - `--resolution`: resampling resolution in metres (default 50)

3. **Bit masking.** `bit_masking` applies each strip's bitmask in parallel to remove bad pixels.

4. **Tide correction.** `tide_correction` reads each strip's acquisition time from its metadata. It computes tides with the **CATS2008_v2023** model on a coarse grid (1 km by default; the model's native resolution is 2 km), interpolates them onto the DEM grid and subtracts them. Floating ice moves up and down with the tide, so this step matters over ice shelves. CATS2008_v2023 can be downloaded from https://www.usap-dc.org/view/dataset/601772. Point `tide_model_dir` to wherever you save it (e.g. `data/initial`).

5. **Co-registration.** `coregistration` aligns each tide-corrected strip to CryoSat-2 POCA points and monthly SWATH elevations from around the acquisition date. It fits a weighted plane (an offset plus a tilt) in two passes, rejecting outliers after the first. If there are too few points for a plane, it falls back to an offset only. The function writes the co-registered strips to `data/interim/region-{region_id}/REMA/co-registered-strips`. It also returns a QC table, saved there as `coreg_quality_control.csv`, with the fit, the RMSEs and a `status` for each strip:
   - `OK`: no flags raised
   - `FLAGGED`: corrected, but at least one flag was raised
   - `UNCORRECTED`: no reference observations over the strip
   - `FAILED`: processing error

   The flags, and the parameters that control them:

   | Flag | Raised when | Parameter (default) |
   |---|---|---|
   | `LOW_POCA_COUNT` | too few POCA points | `min_poca` (10) |
   | `LOW_SWATH_COUNT` | too few SWATH points | `min_swath` (20) |
   | `LOW_FOOTPRINT_COVERAGE` | reference points cover too little of the footprint | `min_footprint_coverage` (0.20) |
   | `POOR_SPATIAL_DISTRIBUTION` | reference points in too few of the footprint's quadrants | `min_covered_quadrants` (3) |
   | `NO_REFERENCE_OBSERVATIONS` | no reference points at all (strip left uncorrected) | |
   | `OFFSET_ONLY` | too few points for a plane, so only an offset was fitted | `min_points_for_plane` (10) |
   | `HIGH_RMSE` | RMSE after correction too high | `max_rmse_m` (10 m) |
   | `RMSE_INCREASED` | the weighted RMSE (the quantity the fit minimises) went up | |
   | `HIGH_REJECTION` | too many points rejected in the second pass, which usually means the DEM itself is bad (e.g. clouds), not just offset | `max_rejected_fraction` (0.5) |

6. **Quality control.** `coreg-qc.ipynb` checks the co-registration in two layers:
   - **Layer 1: overview of all strips**, from the QC table: how often each flag is raised, RMSE before vs. after, the size of the correction (offset, tilt and the largest correction anywhere on the footprint), the offset over time, maps, and how the number of `OK` strips depends on the thresholds. The flags can be re-applied with other thresholds without re-running the co-registration. The layer ends with a shortlist of strips to inspect.
   - **Layer 2: diagnostics for single strips.** For each strip, the reference points and fit are re-created exactly as in `coregistration` (`coreg_reference_points`), and a panel shows the DEM, the residuals before and after on a map, the correction plane, the residuals along and across the strip, and the residual distributions. Use `show_strip(...)` to look at any strip in the notebook. Panels for the shortlist are saved to `co-registered-strips/qc_panels/shortlist`, and panels for all strips can be saved with `save_all = True`.

   - **Layer 3: strip-to-strip adjustment** (step 7): the tiers, strips per year, the agreement between overlapping strips before and after the adjustment, a validation with held-out pairs, and the uncertainty of the corrections.

   Keep the thresholds and fit parameters at the top of `coreg-qc.ipynb` the same as in the `coregistration` and `strip_adjustment` calls in `data-preprocessing.ipynb`.

   Note that all RMSEs in the QC table are computed on the points the plane was fitted to. A large correction is not a problem in itself (some strips are tens of metres off and are corrected cleanly), but a large correction combined with many rejected points usually means a bad DEM.

7. **Strip-to-strip adjustment.** `strip_adjustment` improves on the per-strip co-registration by also using the overlaps between strips. It works on the tide-corrected strips, inside the AOI polygon plus a buffer (`buffer_m`, 10 km), with all strips block-averaged onto a 500 m grid aligned with the SWATH grid. It has three steps:
   1. **Discard bad DEMs**, judged inside the polygon: strips with almost no data there (`LOW_POLYGON_COVERAGE`), or whose residuals against CryoSat-2 have many outliers (`MANY_OUTLIERS`) or a high spread (`HIGH_NMAD`).
   2. **Adjust all strips jointly**: one plane per strip, solved from the CryoSat-2 residuals together with the height differences between strips that overlap within `max_dt_days` (10 days, no correction for ice flow). Each group of observations (per strip and source, and per pair) is weighted by its spread, with its total weight capped at `n_eff` observations to account for spatial correlation, and outliers are rejected. `n_eff = 100` was calibrated with held-out pairs: the predicted uncertainties then match the misfit of pairs left out of the fit.
   3. **Assign tiers.** Strips that disagree with their overlapping strips (`OVERLAP_DISAGREEMENT`) or still fit CryoSat-2 poorly (`POOR_FIT`) are removed one at a time, worst first, and the adjustment is solved again. The remaining strips are:
      - `TIER_A`: tied to CryoSat-2 (directly or through overlaps), correction known to within `max_sigma_m` (2 m) everywhere in the polygon, and agreeing with all overlapping strips to within `max_pair_offset_m` (5 m). Use these for absolute elevations and elevation change.
      - `TIER_B`: a good DEM, but not reliably tied to absolute heights (`tier_b_reason`: `NOT_ANCHORED`, `HIGH_UNCERTAINTY` or `PAIR_OFFSET`). Use these for relative measurements within a strip, such as channel geometry.

   The adjusted strips (cropped to the polygon plus buffer, `*_adj.tif`) and the tables `adjustment_strips.csv` (one row per strip, with the tier, the reasons, the fitted plane and its uncertainty) and `adjustment_pairs.csv` (overlap statistics before and after) are written to `data/interim/region-{region_id}/REMA/adjusted-strips`. The gridded strips are cached in `grid_cache`, so changing the thresholds and running again only takes seconds.

The result is a set of cleaned, tide-corrected, altimetry-aligned REMA strips for the region, sorted into tiers, plus the QC tables and the QC notebook.

## Files

- `data-preprocessing.ipynb`: the step-by-step workflow
- `coreg-qc.ipynb`: quality control of the co-registration
- `preprocessing_tools.py`: functions for AOI handling, bit masking, tide correction, co-registration and the strip-to-strip adjustment
- `qc_tools.py`: plotting functions for the co-registration diagnostics
- `download-REMA-strips.py`: command-line script for downloading (and optionally resampling) the strips
- `environment.yml`: conda environment with the required libraries

## Installation

Create the conda environment with:

```
conda env create -f environment.yml
```
