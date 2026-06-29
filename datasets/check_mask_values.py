import rasterio
import numpy as np

with rasterio.open("data/processed/masks/1_mask.tif") as src:
    mask = src.read(1)

print("Unique values:", np.unique(mask))
print("Foreground pixels:", np.sum(mask == 1))
print("Background pixels:", np.sum(mask == 0))