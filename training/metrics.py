
import torch


class SegmentationMetrics:

    def __init__(self):
        self.reset()

    def reset(self):

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, logits, target):

        pred = torch.argmax(logits, dim=1)

        self.tp += ((pred == 1) & (target == 1)).sum().item()
        self.fp += ((pred == 1) & (target == 0)).sum().item()
        self.fn += ((pred == 0) & (target == 1)).sum().item()
        self.tn += ((pred == 0) & (target == 0)).sum().item()

    def compute(self):

        eps = 1e-7

        precision = self.tp / (self.tp + self.fp + eps)

        recall = self.tp / (self.tp + self.fn + eps)

        dice = (2 * self.tp) / (
            2 * self.tp + self.fp + self.fn + eps
        )

        iou = self.tp / (
            self.tp + self.fp + self.fn + eps
        )

        pixel_acc = (
            self.tp + self.tn
        ) / (
            self.tp + self.tn + self.fp + self.fn + eps
        )

        return {

            "IoU": float(iou),

            "Dice": float(dice),

            "Precision": float(precision),

            "Recall": float(recall),

            "F1": float(dice),

            "PixelAcc": float(pixel_acc),

        }
