import random
import numpy as np
import torch

from torch.utils.data import DataLoader

from training.dataset import TurfDataset
from training.trainer import Trainer

from training.losses import CombinedLoss

from models.segformer import get_model

from configs.config import (
    DEVICE,
    TRAIN_CSV,
    VAL_CSV,
    BATCH_SIZE,
    NUM_WORKERS,
    LR,
    WEIGHT_DECAY,
    EPOCHS,
    SEED,
)


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def main():

    set_seed(SEED)

    print("=" * 60)
    print("SegFormer Training")
    print("=" * 60)

    print(f"Device : {DEVICE}")
    print()

    print("Loading datasets...")
    print()

    train_dataset = TurfDataset(
        TRAIN_CSV,
        train=True,
    )

    val_dataset = TurfDataset(
        VAL_CSV,
        train=False,
    )

    print(f"Train Samples      : {len(train_dataset)}")
    print(f"Validation Samples : {len(val_dataset)}")
    print()

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        drop_last=True,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=NUM_WORKERS > 0,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        drop_last=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=NUM_WORKERS > 0,

    )

    print("Loading SegFormer-B2...")
    print()

    model = get_model()

    criterion = CombinedLoss()

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY,

    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS,

    )

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

    )

    print("=" * 60)
    print("Starting Training")
    print("=" * 60)

    trainer.fit(EPOCHS)


if __name__ == "__main__":

    main()
    