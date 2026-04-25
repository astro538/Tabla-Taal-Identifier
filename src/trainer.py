"""
Training engine for Tabla audio classification models.

Handles:
- Training loop with Mixup augmentation
- Validation with metrics tracking
- Early stopping
- Learning rate scheduling with warmup
- K-fold cross-validation orchestration
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

from src.config import Config


def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("  Using CPU")
    return device


def mixup_data(x, y, alpha=0.3):
    """Apply Mixup augmentation to a batch.
    
    Blends pairs of samples and their labels with a random ratio.
    This regularization technique significantly improves generalization
    on small datasets.
    
    Args:
        x: input batch (B, C, H, W)
        y: label batch (B,)
        alpha: Beta distribution parameter (higher = more mixing)
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 for stability
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * factor)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience=10, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Training engine for a single model."""

    def __init__(self, model, config: Config, device, class_names=None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.class_names = class_names or [str(i) for i in range(config.num_classes)]

        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.training.label_smoothing
        )

        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=config.training.warmup_epochs,
            total_epochs=config.training.epochs,
            min_lr=config.training.min_lr
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            mode="max"
        )

        # Tracking
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": []
        }
        self.best_val_acc = 0.0
        self.best_model_state = None

    def train_epoch(self, dataloader):
        """Train for one epoch with Mixup augmentation."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            # Apply Mixup
            use_mixup = self.config.training.mixup_alpha > 0
            if use_mixup:
                x, y_a, y_b, lam = mixup_data(
                    x, y, self.config.training.mixup_alpha
                )

            # Forward pass
            outputs = self.model(x)

            # Compute loss
            if use_mixup:
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
            else:
                loss = self.criterion(outputs, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for LSTM models)
            if self.config.training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            if use_mixup:
                correct += (lam * predicted.eq(y_a).sum().item()
                            + (1 - lam) * predicted.eq(y_b).sum().item())
            else:
                correct += predicted.eq(y).sum().item()

        avg_loss = total_loss / total
        avg_acc = correct / total
        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, dataloader):
        """Validate the model and return metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(all_labels)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)

        return avg_loss, acc, all_preds, all_labels

    def fit(self, train_loader, val_loader, verbose=True):
        """Full training loop with early stopping and LR scheduling.
        
        Returns:
            dict with training history and best metrics
        """
        epochs = self.config.training.epochs
        model_name = self.model.__class__.__name__

        if verbose:
            pbar = tqdm(range(epochs), desc=f"  {model_name}", leave=False)
        else:
            pbar = range(epochs)

        for epoch in pbar:
            # Update learning rate
            self.scheduler.step(epoch)
            current_lr = self.scheduler.get_lr()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.best_preds = val_preds
                self.best_labels = val_labels

            # Update progress bar
            if verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    "t_loss": f"{train_loss:.3f}",
                    "t_acc": f"{train_acc:.3f}",
                    "v_loss": f"{val_loss:.3f}",
                    "v_acc": f"{val_acc:.3f}",
                    "best": f"{self.best_val_acc:.3f}",
                    "lr": f"{current_lr:.1e}"
                })

            # Early stopping check
            if self.early_stopping(val_acc):
                if verbose:
                    tqdm.write(
                        f"    Early stopping at epoch {epoch + 1}, "
                        f"best val_acc: {self.best_val_acc:.4f}"
                    )
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        return self._compile_results()

    def _compile_results(self):
        """Compile training results into a dictionary."""
        # Compute detailed metrics on best validation predictions
        preds = self.best_preds
        labels = self.best_labels

        precision = precision_score(labels, preds, average="weighted", zero_division=0)
        recall = recall_score(labels, preds, average="weighted", zero_division=0)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(labels, preds)
        report = classification_report(
            labels, preds,
            target_names=self.class_names,
            zero_division=0,
            output_dict=True
        )

        return {
            "accuracy": self.best_val_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "history": self.history,
            "predictions": preds,
            "true_labels": labels,
            "epochs_trained": len(self.history["train_loss"]),
        }

    @torch.no_grad()
    def predict_proba(self, dataloader):
        """Get prediction probabilities for ensemble."""
        self.model.eval()
        all_probs = []
        all_labels = []

        for x, y in dataloader:
            x = x.to(self.device)
            outputs = self.model(x)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(y.numpy())

        return np.concatenate(all_probs, axis=0), np.array(all_labels)

    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        return checkpoint
