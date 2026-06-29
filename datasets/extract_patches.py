import os
import rasterio
from rasterio.windows import Window

# =====================================
# Directories
# =====================================

IMAGE_DIR = "data/processed"
MASK_DIR = "data/processed/masks"

OUT_IMAGE_DIR = "data/processed/patches/images"
OUT_MASK_DIR = "data/processed/patches/masks"

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# =====================================
# Patch Configuration
# =====================================

PATCH_HEIGHT = 256
PATCH_WIDTH = 512

STRIDE_Y = 256
STRIDE_X = 512

patch_id = 0

# =====================================
# Generate Patches
# =====================================

for image_file in sorted(os.listdir(IMAGE_DIR)):

    if not image_file.endswith("_georef.tif"):
        continue

    image_id = image_file.split("_")[0]
    mask_file = f"{image_id}_mask.tif"

    image_path = os.path.join(IMAGE_DIR, image_file)
    mask_path = os.path.join(MASK_DIR, mask_file)

    print(f"\nProcessing {image_file}")

    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:

        width = img_src.width
        height = img_src.height

        for y in range(
            0,
            height - PATCH_HEIGHT + 1,
            STRIDE_Y,
        ):

            for x in range(
                0,
                width - PATCH_WIDTH + 1,
                STRIDE_X,
            ):

                window = Window(
                    col_off=x,
                    row_off=y,
                    width=PATCH_WIDTH,
                    height=PATCH_HEIGHT,
                )

                img_patch = img_src.read(window=window)
                mask_patch = mask_src.read(1, window=window)

                # --------------------------
                # Sanity Check
                # --------------------------

                assert img_patch.shape[1:] == mask_patch.shape, \
                    "Image-Mask size mismatch!"

                transform = img_src.window_transform(window)

                image_profile = img_src.profile.copy()

                image_profile.update(

                    width=PATCH_WIDTH,
                    height=PATCH_HEIGHT,
                    transform=transform,

                )

                mask_profile = mask_src.profile.copy()

                mask_profile.update(

                    width=PATCH_WIDTH,
                    height=PATCH_HEIGHT,
                    transform=transform,

                )

                image_name = f"patch_{patch_id:04d}.tif"
                mask_name = f"patch_{patch_id:04d}.tif"

                image_output = os.path.join(
                    OUT_IMAGE_DIR,
                    image_name,
                )

                mask_output = os.path.join(
                    OUT_MASK_DIR,
                    mask_name,
                )

                with rasterio.open(
                    image_output,
                    "w",
                    **image_profile
                ) as dst:

                    dst.write(img_patch)

                with rasterio.open(
                    mask_output,
                    "w",
                    **mask_profile
                ) as dst:

                    dst.write(mask_patch, 1)

                patch_id += 1

print("\n===================================")
print("Patch Generation Completed")
print("===================================")
print(f"Patch Size : {PATCH_HEIGHT} x {PATCH_WIDTH}")
print(f"Stride     : {STRIDE_Y} x {STRIDE_X}")
print(f"Total Patches : {patch_id}")