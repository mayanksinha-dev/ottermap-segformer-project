import os
from inspect_tiff import TiffFile
import rasterio
from rasterio.transform import from_bounds

INPUT_DIR = "data/raw/images"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in sorted(os.listdir(INPUT_DIR)):

    # Only process original TIFFs
    if not file.lower().endswith((".tif", ".tiff")):
        continue

    # Skip already georeferenced files
    if "_georef" in file.lower():
        print(f"Skipping: {file}")
        continue

    input_path = os.path.join(INPUT_DIR, file)

    # Safe filename
    base_name = os.path.splitext(file)[0]
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_georef.tif"
    )

    print(f"\nProcessing: {file}")

    with TiffFile(input_path) as tif:

        tags = tif.pages[0].tags

        if "ModelTiepointTag" not in tags:
            print(f"❌ No georeference found in {file}")
            continue

        tie = tags["ModelTiepointTag"].value

    xmin = tie[3]
    ymax = tie[4]

    xmax = tie[9]
    ymin = tie[16]

    with rasterio.open(input_path) as src:

        transform = from_bounds(
            xmin,
            ymin,
            xmax,
            ymax,
            src.width,
            src.height,
        )

        profile = src.profile.copy()

        profile.update(
            crs="EPSG:4326",
            transform=transform,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(src.read())

    print(f"✓ Saved -> {output_path}")

print("\nDone!")