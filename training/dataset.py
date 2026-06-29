import rasterio
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import albumentations as A
import cv2

from configs.config import IMAGE_HEIGHT, IMAGE_WIDTH


def get_train_transform():

    return A.Compose([

        A.Resize(
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        ),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.Rotate(
            limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),

        A.RandomBrightnessContrast(p=0.3),

        A.GaussNoise(p=0.2),

        A.GaussianBlur(p=0.2),

        A.Normalize(),

    ])


def get_val_transform():

    return A.Compose([

        A.Resize(
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        ),

        A.Normalize(),

    ])


class TurfDataset(Dataset):

    def __init__(
        self,
        csv_file,
        train=True,
    ):

        self.df = pd.read_csv(csv_file)

        self.transform = (
            get_train_transform()
            if train
            else get_val_transform()
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        image_path = self.df.iloc[idx]["image"]
        mask_path = self.df.iloc[idx]["mask"]

        # ----------------------------
        # Read Image with Rasterio
        # ----------------------------

        with rasterio.open(image_path) as src:

            image = src.read([1, 2, 3])

        image = np.transpose(image, (1, 2, 0))

        image = image.astype(np.float32)

        # Scale if image is uint16
        if image.max() > 255:

            image = image / image.max() * 255

        image = image.astype(np.uint8)

        # ----------------------------
        # Read Mask with Rasterio
        # ----------------------------

        with rasterio.open(mask_path) as src:

            mask = src.read(1)

        mask = mask.astype(np.uint8)

        transformed = self.transform(
            image=image,
            mask=mask,
        )

        image = transformed["image"]
        mask = transformed["mask"]

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()

        mask = torch.from_numpy(
            mask
        ).long()

        return image, mask