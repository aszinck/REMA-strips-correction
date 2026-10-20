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

5. **Co-registration.** `coregistration` aligns each tide-corrected strip to CryoSat-2 POCA points and monthly SWATH elevations from around the acquisition date. It fits a weighted plane (an offset plus a tilt) in two passes, rejecting outliers after the first. If there are too few points for a plane, it falls back to an offset only. Each strip goes through QC checks: minimum number of POCA/SWATH points, footprint coverage, number of covered quadrants and maximum RMSE. The function writes the co-registered strips to `data/interim/region-{region_id}/REMA/co-registered-strips` and returns a QC table with a `status` for each strip.

The result is a set of cleaned, tide-corrected, altimetry-aligned REMA strips for the region, plus the QC table.

## Files

- `data-preprocessing.ipynb`: the step-by-step workflow
- `preprocessing_tools.py`: functions for AOI handling, bit masking, tide correction and co-registration
- `download-REMA-strips.py`: command-line script for downloading (and optionally resampling) the strips
- `environment.yml`: conda environment with the required libraries

## Installation

Create the conda environment with:

```
conda env create -f environment.yml
```
