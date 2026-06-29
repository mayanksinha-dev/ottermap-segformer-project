import torch
import torch.nn.functional as F

from tqdm import tqdm

from configs.config import (
    DEVICE,
    CHECKPOINT_DIR,
    MAX_GRAD_NORM,
    USE_AMP,
    PATIENCE,
)

from training.metrics import SegmentationMetrics


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
    ):

        CHECKPOINT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.device = DEVICE

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=USE_AMP,
        )

        self.metrics = SegmentationMetrics()

        self.best_iou = 0.0

        self.no_improve = 0

    
    def train_one_epoch(self):
    
        self.model.train()
    
        running_loss = 0.0
    
        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        )
    
        for images, masks in progress:
    
            images = images.to(self.device)
            masks = masks.to(self.device)
    
            self.optimizer.zero_grad(set_to_none=True)
    
            with torch.amp.autocast(
                device_type="cuda",
                enabled=USE_AMP,
            ):
    
                outputs = self.model(
                    pixel_values=images
                )
    
                logits = outputs.logits
    
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
    
                loss = self.criterion(
                    logits,
                    masks,
                )
    
            self.scaler.scale(loss).backward()
    
            self.scaler.unscale_(self.optimizer)
    
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                MAX_GRAD_NORM,
            )
    
            self.scaler.step(self.optimizer)
    
            self.scaler.update()
    
            running_loss += loss.item()
    
            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )
    
        avg_loss = running_loss / len(self.train_loader)
    
        return avg_loss
    
    
    @torch.no_grad()
    def validate(self):
    
        self.model.eval()
    
        self.metrics.reset()
    
        running_loss = 0.0
    
        progress = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
        )
    
        for images, masks in progress:
    
            images = images.to(self.device)
            masks = masks.to(self.device)
    
            with torch.amp.autocast(
                device_type="cuda",
                enabled=USE_AMP,
            ):
    
                outputs = self.model(
                    pixel_values=images
                )
    
                logits = outputs.logits
    
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
    
                loss = self.criterion(
                    logits,
                    masks,
                )
    
            running_loss += loss.item()
    
            self.metrics.update(
                logits,
                masks,
            )
    
            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )
    
        avg_loss = running_loss / len(self.val_loader)
    
        metrics = self.metrics.compute()
    
        return avg_loss, metrics
    
    
    def save_checkpoint(
        self,
        epoch,
        iou,
    ):
    
        checkpoint = {
    
            "epoch": epoch,
    
            "model_state_dict": self.model.state_dict(),
    
            "optimizer_state_dict": self.optimizer.state_dict(),
    
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
    
            "best_iou": self.best_iou,
    
        }
    
        # Always save latest checkpoint
        torch.save(
            checkpoint,
            CHECKPOINT_DIR / "last_model.pth",
        )
    
        # Save best checkpoint
        if iou > self.best_iou:
    
            self.best_iou = iou
    
            checkpoint["best_iou"] = self.best_iou
    
            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "best_model.pth",
            )
    
            self.no_improve = 0
    
            print(
                f"\n✅ New best model saved! IoU = {iou:.4f}"
            )
    
        else:
    
            self.no_improve += 1

    
    def fit(self, epochs):
    
        print("=" * 60)
        print("Starting Training...")
        print("=" * 60)
    
        for epoch in range(epochs):
    
            print()
            print("=" * 60)
            print(f"Epoch {epoch + 1}/{epochs}")
            print("=" * 60)
    
            # -----------------------
            # Train
            # -----------------------
    
            train_loss = self.train_one_epoch()
    
            # -----------------------
            # Validate
            # -----------------------
    
            val_loss, metrics = self.validate()
    
            # -----------------------
            # Scheduler
            # -----------------------
    
            if self.scheduler is not None:
                self.scheduler.step()
    
            # -----------------------
            # Save Model
            # -----------------------
    
            self.save_checkpoint(
                epoch=epoch,
                iou=metrics["IoU"],
            )
    
            # -----------------------
            # Print Results
            # -----------------------
    
            print()
    
            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {val_loss:.4f}")
    
            print()
    
            print("Validation Metrics")
            print("-" * 30)
    
            for name, value in metrics.items():
    
                print(f"{name:<12}: {value:.4f}")
    
            if self.scheduler is not None:
    
                print()
    
                print(
                    f"Learning Rate : {self.optimizer.param_groups[0]['lr']:.6f}"
                )
    
            # -----------------------
            # Early Stopping
            # -----------------------
    
            if self.no_improve >= PATIENCE:
    
                print()
                print("=" * 60)
                print("Early stopping triggered.")
                print("=" * 60)
    
                break
    
        print()
        print("=" * 60)
        print("Training Finished")
        print("=" * 60)