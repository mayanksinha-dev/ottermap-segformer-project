import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show

IMAGE = "data/processed/1_georef.tif"
LABEL = "data/raw/labels/1.geojson"

with rasterio.open(IMAGE) as src:
    fig, ax = plt.subplots(figsize=(12, 12))

    show(src, ax=ax)

    gdf = gpd.read_file(LABEL)

    # Convert labels to image CRS
    gdf = gdf.to_crs(src.crs)

    gdf.boundary.plot(
        ax=ax,
        edgecolor="red",
        linewidth=2
    )

    plt.title("Overlay Check")
    plt.show()