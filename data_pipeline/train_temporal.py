"""
train_temporal.py

Training loop for the LSTM+RNN temporal fusion model, operating on
cached CNN embeddings (from extract_embeddings.py) rather than raw
frames — this is what makes each epoch fast despite the added
architectural complexity, since Xception's expensive forward pass
already happened once during caching, not on every epoch here.

Usage:
    python train_temporal.py --embeddings_root embeddings --epochs 15
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sequence_dataset import VideoSequenceDataset, build_sequence_splits, collate_sequences
from temporal_model import TemporalFusionModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_embedding_dim(sample_path: Path) -> int:
    """Peek at one cached embedding file to determine the feature dimension automatically."""
    import numpy as np
    sample = np.load(sample_path)
    return sample.shape[1]


def run_one_epoch(model, dataloader, criterion, optimizer, device, is_training: bool, phase_name: str) -> dict:
    """Same pattern as train.py's run_one_epoch, adapted for the (sequence, seq_len, label) batch shape."""
    from tqdm import tqdm

    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    progress_bar = tqdm(dataloader, desc=f"  {phase_name}", unit="batch", leave=False)

    with context:
        for sequences, seq_lens, labels in progress_bar:
            sequences, labels = sequences.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(sequences, seq_lens)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                "loss": f"{total_loss / total:.4f}",
                "acc": f"{correct / total:.4f}",
            })

    return {
        "loss": total_loss / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def train_temporal_model(
    embeddings_root: str,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    dropout: float = 0.3,
    early_stopping_patience: int = 3,
    checkpoint_dir: str = "checkpoints",
):
    """
    Full training run for the LSTM+RNN fusion model.

    Note: learning_rate default here (5e-4) is higher than train.py's
    CNN fine-tuning rate (5e-5) — this model is trained fully from
    scratch (unlike the pretrained CNN backbones), so it needs a
    larger learning rate to make meaningful progress, same as any
    from-scratch architecture.
    """
    device = get_device()
    print(f"Using device: {device}")

    train_samples, val_samples, test_samples = build_sequence_splits(embeddings_root)

    if not train_samples:
        raise ValueError(
            "No training samples found. Did extract_embeddings.py run successfully "
            "and produce .npy files under the given embeddings_root?"
        )

    embedding_dim = infer_embedding_dim(train_samples[0][0])
    print(f"Detected embedding dimension: {embedding_dim}")

    train_dataset = VideoSequenceDataset(train_samples)
    val_dataset = VideoSequenceDataset(val_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sequences, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sequences, num_workers=2
    )

    print(f"Train batches per epoch: {len(train_loader)} | Val batches per epoch: {len(val_loader)}")

    model = TemporalFusionModel(embedding_dim=embedding_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        print(f"Epoch {epoch}/{epochs} starting...")
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, True, "train")
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device, False, "val")

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"train_loss: {train_metrics['loss']:.4f} train_acc: {train_metrics['accuracy']:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / "temporal_fusion_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best val accuracy ({best_val_accuracy:.4f}) — saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s) (best so far: {best_val_accuracy:.4f})")

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}/{epochs}.")
                break

    print(f"\nTraining done. Best val accuracy: {best_val_accuracy:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM+RNN temporal fusion model.")
    parser.add_argument("--embeddings_root", required=True, help="Path to cached embeddings (output of extract_embeddings.py).")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    train_temporal_model(
        embeddings_root=args.embeddings_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
