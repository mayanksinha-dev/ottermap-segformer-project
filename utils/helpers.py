import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_directory(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def get_device():
    """
    Return GPU if available.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu")