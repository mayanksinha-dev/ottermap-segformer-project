from torch.utils.data import DataLoader

from configs.config import TRAIN_CSV
from training.dataset import TurfDataset


def main():

    print("=" * 50)
    print("Loading Training Dataset")
    print("=" * 50)

    dataset = TurfDataset(
        TRAIN_CSV,
        train=True,
    )

    print(f"Dataset Size : {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    images, masks = next(iter(loader))

    print()
    print("Image Shape :", images.shape)
    print("Mask Shape  :", masks.shape)
    print()

    print("Image dtype :", images.dtype)
    print("Mask dtype  :", masks.dtype)
    print()

    print("Min Pixel :", images.min().item())
    print("Max Pixel :", images.max().item())

    print("\nDataset Loaded Successfully ✅")


if __name__ == "__main__":
    main()