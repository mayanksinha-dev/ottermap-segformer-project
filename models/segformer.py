import torch

from transformers import (
    SegformerForSemanticSegmentation,
)


MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"


def get_model():

    model = SegformerForSemanticSegmentation.from_pretrained(

        MODEL_NAME,

        num_labels=2,

        ignore_mismatched_sizes=True,

    )

    model.config.num_labels = 2

    model.config.id2label = {

        0: "background",

        1: "turf",

    }

    model.config.label2id = {

        "background": 0,

        "turf": 1,

    }

    print("=" * 60)
    print("SegFormer Model Loaded")
    print("=" * 60)
    print(f"Backbone   : {MODEL_NAME}")
    print(f"Classes    : {model.config.num_labels}")
    print()

    return model


if __name__ == "__main__":

    model = get_model()

    model.eval()

    x = torch.randn(

        2,

        3,

        256,

        512,

    )

    with torch.no_grad():

        output = model(

            pixel_values=x

        )

    print(model.__class__.__name__)

    print(f"Input Shape  : {x.shape}")

    print(f"Output Shape : {output.logits.shape}")

    assert output.logits.shape[1] == 2

    print()

    print("✅ SegFormer test passed.")