from pathlib import Path

path = Path("training/trainer.py")

text = path.read_text()

text = text.replace(
    "for images, masks in self.val_loader:",
    'loop = tqdm(self.val_loader, desc="Validation")\n\n        for images, masks in loop:'
)

path.write_text(text)

print("trainer.py updated")
