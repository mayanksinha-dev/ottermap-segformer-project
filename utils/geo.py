import geopandas as gpd
from shapely.geometry import shape
import rasterio
from rasterio.features import shapes


def mask_to_polygons(mask, transform):
    """
    Convert a binary mask to Shapely polygons.
    """
    polygons = []

    for geom, value in shapes(mask.astype("uint8"), transform=transform):
        if value == 1:
            polygons.append(shape(geom))

    return polygons


def save_geojson(polygons, output_path, crs="EPSG:4326"):
    """
    Save polygons as a GeoJSON file.
    """
    gdf = gpd.GeoDataFrame(
        geometry=polygons,
        crs=crs
    )

    gdf.to_file(output_path, driver="GeoJSON")


def mask_file_to_geojson(mask_path, output_path):
    """
    Convert a raster mask directly into GeoJSON.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        polygons = mask_to_polygons(mask, src.transform)
        save_geojson(polygons, output_path, src.crs)