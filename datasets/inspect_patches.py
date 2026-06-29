import os
import random

import rasterio
import matplotlib.pyplot as plt

IMAGE_DIR = "data/processed/patches/images"
MASK_DIR = "data/processed/patches/masks"

patch = random.choice(sorted(os.listdir(IMAGE_DIR)))

with rasterio.open(os.path.join(IMAGE_DIR, patch)) as src:
    image = src.read([1, 2, 3]).transpose(1, 2, 0)

with rasterio.open(os.path.join(MASK_DIR, patch)) as src:
    mask = src.read(1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Image Shape :", image.shape)
print("Mask Shape  :", mask.shape)