import argparse
import os
import subprocess
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

parser = argparse.ArgumentParser(description="Download and process REMA strips.")

parser.add_argument(
    "--region_id",
    type=str,
    required=True,
    help="Region ID, e.g. '05'"
)

parser.add_argument(
    "--project_crs",
    type=str,
    default="EPSG:3031",
    help="Target CRS, e.g. 'EPSG:3031'"
)

parser.add_argument(
    "--n_threads",
    type=int,
    default=8,
    help="Number of parallel threads"
)

parser.add_argument(
    "--projectDir",
    type=str,
    required=True,
    help="Base project directory"
)

parser.add_argument(
    "--resample",
    type=str,
    choices=["true", "false"],
    default="true",
    help="Whether to resample REMA strips using GDAL (true/false)"
)

parser.add_argument(
    "--resolution",
    type=float,
    default=50,
    help="Target resolution in metres when resampling, e.g. 50"
)

args = parser.parse_args()

# Assign to variables used below
region_id = args.region_id
project_crs = args.project_crs
n_threads = args.n_threads
projectDir = args.projectDir
resample = args.resample.lower() == "true"
resolution = args.resolution

downloader = "aria2c"  # or "wget"

baseDir = f"{projectDir}/data/initial/REMA/region-{region_id}/"
demDir = os.path.join(baseDir, "REMA_strips")
bitDir = os.path.join(baseDir, "REMA_bitmask")
metaDir = os.path.join(baseDir, "metadataFiles")
tmpDir = os.path.join(baseDir, "tmp")

for d in [demDir, bitDir, metaDir, tmpDir]:
    os.makedirs(d, exist_ok=True)


# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def process_file(file_url):
    """Download, extract, optionally resample, and clean up one DEM strip."""

    file_name = os.path.basename(file_url)
    tar_path = os.path.join(tmpDir, file_name)
    base_name = file_name.replace(".tar.gz", "").replace(".tar", "")

    # --------------------------------------------------
    # Define output filenames
    # --------------------------------------------------

    if resample:
        # Avoid filenames such as "_50.0m.tif"
        if resolution.is_integer():
            res_string = str(int(resolution))
        else:
            res_string = str(resolution)

        demFile = os.path.join(
            demDir,
            f"{base_name}_dem_{res_string}m.tif"
        )
        bitFile = os.path.join(
            bitDir,
            f"{base_name}_bitmask_{res_string}m.tif"
        )

    else:
        demFile = os.path.join(
            demDir,
            f"{base_name}_dem.tif"
        )
        bitFile = os.path.join(
            bitDir,
            f"{base_name}_bitmask.tif"
        )

    # Skip if both files already exist
    if os.path.exists(demFile) and os.path.exists(bitFile):
        return f"✅ Skipped {file_name} (already processed)"

    # --------------------------------------------------
    # Download
    # --------------------------------------------------

    try:
        if downloader == "aria2c":
            cmd = [
                "aria2c",
                "-x", "8",
                "-s", "8",
                "-d", tmpDir,
                "-o", file_name,
                file_url
            ]
        else:
            cmd = [
                "wget",
                "-q",
                "-O", tar_path,
                file_url
            ]

        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError:
        return f"❌ Download failed: {file_name}"

    # --------------------------------------------------
    # Extract
    # --------------------------------------------------

    try:
        subprocess.run(
            ["tar", "-xf", tar_path, "-C", tmpDir],
            check=True
        )

    except subprocess.CalledProcessError:
        return f"❌ Extraction failed: {file_name}"

    # --------------------------------------------------
    # Move metadata and process raster files
    # --------------------------------------------------

    try:

        # Move metadata
        meta_src = os.path.join(
            tmpDir,
            f"{base_name}_mdf.txt"
        )

        if os.path.exists(meta_src):
            meta_dst = os.path.join(
                metaDir,
                f"{base_name}_mdf.txt"
            )
            os.replace(meta_src, meta_dst)

        # Original extracted files
        dem_src = os.path.join(
            tmpDir,
            f"{base_name}_dem.tif"
        )

        bit_src = os.path.join(
            tmpDir,
            f"{base_name}_bitmask.tif"
        )

        # --------------------------------------------------
        # Option 1: Resample with GDAL
        # --------------------------------------------------

        if resample:

            if os.path.exists(dem_src):
                subprocess.run([
                    "gdalwarp",
                    "-tr",
                    str(resolution),
                    str(resolution),
                    "-r",
                    "near",
                    dem_src,
                    demFile
                ], check=True)

            if os.path.exists(bit_src):
                subprocess.run([
                    "gdalwarp",
                    "-tr",
                    str(resolution),
                    str(resolution),
                    "-r",
                    "near",
                    bit_src,
                    bitFile
                ], check=True)

        # --------------------------------------------------
        # Option 2: Keep original REMA resolution
        # --------------------------------------------------

        else:

            if os.path.exists(dem_src):
                os.replace(dem_src, demFile)

            if os.path.exists(bit_src):
                os.replace(bit_src, bitFile)

    except subprocess.CalledProcessError:
        return f"❌ GDAL processing failed: {file_name}"

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    for f in os.listdir(tmpDir):

        if f.startswith(base_name) or f == "index":

            path = os.path.join(tmpDir, f)

            if os.path.isdir(path):
                subprocess.run(["rm", "-rf", path])
            else:
                os.remove(path)

    if os.path.exists(tar_path):
        os.remove(tar_path)

    if resample:
        return f"✅ Done: {file_name} → {resolution:g} m"
    else:
        return f"✅ Done: {file_name} → original resolution"


# --------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------

if __name__ == "__main__":

    print("🔹 Loading AOI and DEM index...")

    region = gpd.read_file(
        f"{projectDir}/data/initial/regions/region-{region_id}.shp"
    )

    strips = gpd.read_file(
        f"{projectDir}/data/initial/REMA/REMA_Strip_Index_s2s041.shp"
    ).to_crs(project_crs)

    strips_region = gpd.clip(strips, region)

    strips_region = strips_region[
        (strips_region.sensor1 != "GEO1")
        & (strips_region.sensor2 != "GEO1")
    ]

    urls = strips_region.fileurl.tolist()

    print(f"🔹 Found {len(urls)} REMA strips")

    if resample:
        print(f"🔹 Resampling enabled: {resolution:g} m")
    else:
        print("🔹 Resampling disabled: keeping original resolution")

    print(
        f"🚀 Starting parallel processing with "
        f"{n_threads} threads..."
    )

    with ThreadPoolExecutor(max_workers=n_threads) as executor:

        futures = {
            executor.submit(process_file, url): url
            for url in urls
        }

        for future in as_completed(futures):
            print(future.result())

    print("🏁 All downloads completed.")