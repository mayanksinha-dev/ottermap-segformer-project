
from pathlib import Path

from inference.polygonize import polygonize


MASK_DIR = Path("outputs/predictions")

OUTPUT_DIR = Path("outputs/geojson")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

masks = sorted(
    MASK_DIR.glob("*.tif")
)

print(f"Found {len(masks)} masks\n")

total_polygons = 0

for i, mask in enumerate(masks):

    out = OUTPUT_DIR / f"{mask.stem}.geojson"

    n = polygonize(
        mask,
        out,
    )

    total_polygons += n

    print(
        f"[{i+1}/{len(masks)}] "
        f"{mask.name} -> {n} polygons"
    )

print()

print("=" * 60)

print(f"GeoJSON Files : {len(masks)}")

print(f"Total Polygons : {total_polygons}")

print("=" * 60)
