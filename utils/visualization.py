import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_mask(mask, title="Mask"):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def overlay_mask(image, mask, alpha=0.4):
    overlay = image.copy()

    overlay[mask == 1] = [255, 0, 0]

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(overlay, alpha=alpha)
    plt.axis("off")
    plt.show()


def compare_results(image, prediction):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(image)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(prediction, cmap="gray")
    ax[1].set_title("Prediction")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()