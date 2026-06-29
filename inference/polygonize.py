
from pathlib import Path

import geopandas as gpd
import rasterio

from rasterio.features import shapes
from shapely.geometry import shape


def polygonize(mask_path, output_path):

    with rasterio.open(mask_path) as src:

        mask = src.read(1)

        transform = src.transform

        crs = src.crs

    polygons = []

    for geom, value in shapes(mask, transform=transform):

        if value == 1:

            polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame(
        geometry=polygons,
        crs=crs,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gdf.to_file(
        output_path,
        driver="GeoJSON",
    )

    return len(gdf)
