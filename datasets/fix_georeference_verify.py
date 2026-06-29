import os
import rasterio

for file in sorted(os.listdir("data/processed")):
    if file.endswith(".tif"):
        with rasterio.open(os.path.join("data/processed", file)) as src:
            print(file)
            print("CRS:", src.crs)
            print("Transform:", src.transform)
            print("-" * 50)