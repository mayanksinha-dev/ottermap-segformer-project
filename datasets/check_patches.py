import os
import rasterio
import numpy as np

MASK_DIR = "data/processed/patches/masks"

total = 0
positive = 0

for file in os.listdir(MASK_DIR):

    with rasterio.open(os.path.join(MASK_DIR, file)) as src:
        mask = src.read(1)

    total += 1

    if np.any(mask == 1):
        positive += 1

print("Total patches :", total)
print("Positive patches :", positive)
print("Negative patches :", total - positive)