import torch

from training.losses import CombinedLoss

criterion = CombinedLoss()

# Batch of 4 images
logits = torch.randn(
    4,
    2,
    256,
    512,
)

mask = torch.randint(
    0,
    2,
    (
        4,
        256,
        512,
    ),
)

loss = criterion(
    logits,
    mask,
)

print("=" * 40)
print("Loss Test")
print("=" * 40)
print(f"Loss : {loss.item():.6f}")

assert torch.isfinite(loss), "Loss is NaN or Inf"

print("✅ Loss function works correctly.")