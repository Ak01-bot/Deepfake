"""
train.py

Training loop skeleton for fine-tuning a pretrained CNN backbone
(starting with Xception) for real/fake binary classification.

This is written and syntax-checked so Week 2 starts on actually running
and tuning training, not on writing this infrastructure from scratch.

Usage:
    python train.py --faces_root faces --backbone xception --epochs 10
"""

import argparse
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FaceDataset, build_dataset_splits, get_eval_transform, get_train_transform


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(backbone_name: str, device: torch.device) -> nn.Module:
    """
    Load a pretrained backbone via timm and swap the classifier head
    for 2-class (real/fake) output. This is the transfer-learning setup
    validated in the earlier sanity check (xception/efficientnet_b0/resnet50
    all confirmed loading with pretrained weights and correct output shape).
    """
    model = timm.create_model(backbone_name, pretrained=True, num_classes=2)
    return model.to(device)


def run_one_epoch(model, dataloader, criterion, optimizer, device, is_training: bool) -> dict:
    """
    Run a single pass over the given dataloader.

    If is_training is True, this updates model weights (train mode).
    If False, this only evaluates (val/test mode, no gradient updates).

    Returns:
        Dict with average loss and accuracy for this pass.
    """
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def train_model(
    faces_root: str,
    backbone_name: str = "xception",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
):
    """
    Full training run: builds splits, dataloaders, model, and runs the
    train/val loop for the given number of epochs, saving the best
    checkpoint (by validation accuracy) along the way.
    """
    device = get_device()
    print(f"Using device: {device}")

    # --- Data ---
    train_samples, val_samples, test_samples = build_dataset_splits(faces_root)

    train_dataset = FaceDataset(train_samples, transform=get_train_transform())
    val_dataset = FaceDataset(val_samples, transform=get_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- Model, loss, optimizer ---
    model = build_model(backbone_name, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0

    # --- Epoch loop ---
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device, is_training=False)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"train_loss: {train_metrics['loss']:.4f} train_acc: {train_metrics['accuracy']:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            checkpoint_path = checkpoint_dir / f"{backbone_name}_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best val accuracy ({best_val_accuracy:.4f}) — saved to {checkpoint_path}")

    print(f"\nTraining done. Best val accuracy: {best_val_accuracy:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained CNN backbone for deepfake detection.")
    parser.add_argument("--faces_root", required=True, help="Path to the 'faces' folder (output of mtcnn_align.py).")
    parser.add_argument("--backbone", default="xception", help="timm model name (default: xception).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    train_model(
        faces_root=args.faces_root,
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
