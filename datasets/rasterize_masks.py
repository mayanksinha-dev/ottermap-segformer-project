import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon, MultiPolygon

IMAGE_DIR = "data/processed"
LABEL_DIR = "data/raw/labels"
MASK_DIR = "data/processed/masks"

os.makedirs(MASK_DIR, exist_ok=True)


def remove_holes(geom):
    """
    Remove interior rings (holes) from polygons.
    """
    if geom is None:
        return None

    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)

    if geom.geom_type == "MultiPolygon":
        return MultiPolygon(
            [Polygon(poly.exterior) for poly in geom.geoms]
        )

    return geom


for image_file in sorted(os.listdir(IMAGE_DIR)):

    if not image_file.endswith("_georef.tif"):
        continue

    image_id = image_file.split("_")[0]

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    label_path = os.path.join(
        LABEL_DIR,
        f"{image_id}.geojson"
    )

    print(f"\nProcessing {image_file}")

    with rasterio.open(image_path) as src:

        gdf = gpd.read_file(label_path)

        # Match CRS
        gdf = gdf.to_crs(src.crs)

        # Fix invalid geometries
        gdf["geometry"] = gdf.geometry.buffer(0)

        # Remove polygon holes
        gdf["geometry"] = gdf.geometry.apply(remove_holes)

        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry if geom is not None],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        profile = src.profile.copy()

        profile.update(
            count=1,
            dtype=rasterio.uint8,
            compress="lzw",
        )

        output_path = os.path.join(
            MASK_DIR,
            f"{image_id}_mask.tif"
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)

    print(f"✓ Saved {output_path}")

print("\nDone!")