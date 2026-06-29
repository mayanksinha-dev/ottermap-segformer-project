import matplotlib.pyplot as plt
import rasterio

fig, ax = plt.subplots(1, 2, figsize=(15, 8))

with rasterio.open("data/processed/1_georef.tif") as src:
    image = src.read([1, 2, 3]).transpose(1, 2, 0)

with rasterio.open("data/processed/masks/1_mask.tif") as src:
    mask = src.read(1)

ax[0].imshow(image)
ax[0].set_title("Image")

ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Mask")

plt.show()