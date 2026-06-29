
import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from models.segformer import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


def preprocess(image):

    image = image.astype(np.float32)

    image = image / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    image = (image - mean) / std

    image = torch.from_numpy(image).float()

    image = image.permute(2, 0, 1)

    image = image.unsqueeze(0)

    return image


def predict(model, image):

    image = preprocess(image).to(DEVICE)

    with torch.no_grad():

        output = model(pixel_values=image)

        logits = output.logits

        logits = F.interpolate(
            logits,
            size=(256, 512),
            mode="bilinear",
            align_corners=False,
        )

        pred = torch.argmax(logits, dim=1)

    return pred.squeeze().cpu().numpy().astype(np.uint8)


def create_overlay(image, mask):

    overlay = image.copy()

    overlay[mask == 1] = [255, 0, 0]

    overlay = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0,
    )

    return overlay


def process_image(image_path, model):

    image_path = Path(image_path)

    with rasterio.open(image_path) as src:

        image = src.read([1, 2, 3])

        profile = src.profile.copy()

    image = np.moveaxis(image, 0, -1)

    prediction = predict(model, image)

    import cv2

    prediction = cv2.resize(
        prediction.astype("uint8"),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = create_overlay(image, prediction)

    pred_dir = Path("outputs/predictions")
    overlay_dir = Path("outputs/overlays")

    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"{image_path.stem}_mask.tif"

    profile.update(
        count=1,
        dtype=rasterio.uint8,
    )

    with rasterio.open(
        pred_path,
        "w",
        **profile,
    ) as dst:

        dst.write(prediction, 1)

    overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR,
        ),
    )

    print("Prediction :", pred_path)
    print("Overlay    :", overlay_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Loading SegFormer...")
    print("=" * 60)

    model = get_model()

    model = load_checkpoint(
        model,
        args.checkpoint,
    )

    print("Model loaded successfully.\n")

    process_image(
        args.image,
        model,
    )

    print("\nInference completed successfully.")


if __name__ == "__main__":
    main()
