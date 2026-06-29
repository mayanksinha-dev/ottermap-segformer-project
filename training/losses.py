import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0,
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets,
    ):

        if logits.shape[1] != 2:
            raise ValueError(
                "DiceLoss expects logits with 2 classes."
            )

        probs = F.softmax(
            logits,
            dim=1,
        )

        probs = probs[:, 1]

        targets = targets.float()

        probs = probs.contiguous().view(-1)

        targets = targets.contiguous().view(-1)

        intersection = (probs * targets).sum()

        denominator = (
            probs.sum()
            + targets.sum()
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice


class CombinedLoss(nn.Module):

    def __init__(
        self,
        dice_weight=0.5,
        ce_weight=0.5,
        ignore_index=-100,
    ):

        super().__init__()

        self.dice = DiceLoss()

        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
        )

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(
        self,
        logits,
        targets,
    ):

        dice_loss = self.dice(
            logits,
            targets,
        )

        ce_loss = self.ce(
            logits,
            targets,
        )

        total_loss = (

            self.dice_weight * dice_loss

            +

            self.ce_weight * ce_loss

        )

        return total_loss