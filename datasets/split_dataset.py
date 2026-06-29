import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)

IMAGE_DIR = "data/processed/patches/images"
MASK_DIR = "data/processed/patches/masks"

images = sorted(os.listdir(IMAGE_DIR))

records = []

for image_name in images:

    records.append({

        "image": os.path.join(
            IMAGE_DIR,
            image_name,
        ).replace("\\", "/"),

        "mask": os.path.join(
            MASK_DIR,
            image_name,
        ).replace("\\", "/")

    })

df = pd.DataFrame(records)

train_df, val_df = train_test_split(

    df,

    test_size=0.2,

    random_state=42,

    shuffle=True,

)

train_df.to_csv(
    "data/train.csv",
    index=False,
)

val_df.to_csv(
    "data/val.csv",
    index=False,
)

print("=" * 40)
print("Dataset Split Completed")
print("=" * 40)
print("Total :", len(df))
print("Train :", len(train_df))
print("Val   :", len(val_df))