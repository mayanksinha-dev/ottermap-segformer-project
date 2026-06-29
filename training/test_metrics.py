import torch

from training.metrics import SegmentationMetrics

metrics = SegmentationMetrics()

logits = torch.randn(
    2,
    2,
    256,
    512,
)

mask = torch.randint(
    0,
    2,
    (
        2,
        256,
        512,
    ),
)

metrics.reset()

metrics.update(
    logits,
    mask,
)

result = metrics.compute()

print("=" * 40)
print("Metrics Test")
print("=" * 40)

for k, v in result.items():
    print(f"{k:<12}: {v:.4f}")

print("\n✅ Metrics working correctly.")