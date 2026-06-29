
from pathlib import Path

from inference.inference import (
    get_model,
    load_checkpoint,
    process_image,
    DEVICE,
)

print("=" * 60)
print("Loading model once...")
print("=" * 60)

model = get_model().to(DEVICE)

model = load_checkpoint(
    model,
    "checkpoints/best_model.pth",
)

print("Model loaded.\n")

IMAGE_DIR = Path("data/processed/patches/images")

images = sorted(IMAGE_DIR.glob("*.tif"))

print(f"Found {len(images)} images\n")

for i, image in enumerate(images):

    print(f"[{i+1}/{len(images)}] {image.name}")

    process_image(
        image,
        model,
    )

print("\nInference completed.")
