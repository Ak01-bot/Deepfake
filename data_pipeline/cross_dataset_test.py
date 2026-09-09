"""
cross_dataset_test.py

Cross-dataset generalization test: evaluates the trained model on
Celeb-DF — a completely different dataset (different subjects,
different source videos, different manipulation pipeline) than
FaceForensics++, which the model was actually trained on.

This answers a broader version of the Week 7 question than the
faceshifter_holdout test: not just "does it generalize to an unseen
manipulation technique within the same dataset," but "does it
generalize to an entirely different dataset altogether."

Expects two separate embedding folders, one per class (produced by
running extract_embeddings_holdout.py separately on each):
    --real_embeddings_root  : embeddings from Celeb-DF's real videos
    --fake_embeddings_root  : embeddings from Celeb-DF's fake videos

Usage:
    python cross_dataset_test.py --real_embeddings_root celebdf_embeddings/real \\
        --fake_embeddings_root celebdf_embeddings/fake \\
        --checkpoint checkpoints/temporal_attention.pth --use_attention
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from sequence_dataset import VideoSequenceDataset, collate_sequences
from temporal_model import TemporalFusionModel, TemporalFusionAttentionModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_embedding_dim(sample_path) -> int:
    sample = np.load(sample_path)
    return sample.shape[1]


def load_samples(embeddings_root: str, label: int) -> list:
    root = Path(embeddings_root)
    return [(f, label) for f in sorted(root.glob("*.npy"))]


def run_cross_dataset_test(
    real_embeddings_root: str,
    fake_embeddings_root: str,
    checkpoint_path: str,
    batch_size: int = 16,
    lstm_hidden_size: int = 256,
    lstm_num_layers: int = 2,
    rnn_hidden_size: int = 256,
    rnn_num_layers: int = 1,
    use_attention: bool = False,
):
    device = get_device()
    print(f"Using device: {device}")

    real_samples = load_samples(real_embeddings_root, label=0)
    fake_samples = load_samples(fake_embeddings_root, label=1)
    print(f"Celeb-DF real videos: {len(real_samples)}")
    print(f"Celeb-DF fake videos: {len(fake_samples)}")

    combined_samples = real_samples + fake_samples
    if not combined_samples:
        raise ValueError("No samples found in either embeddings folder — check the paths.")

    print(f"Combined cross-dataset test set: {len(combined_samples)} videos\n")

    embedding_dim = infer_embedding_dim(combined_samples[0][0])

    test_dataset = VideoSequenceDataset(combined_samples)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sequences, num_workers=2
    )

    ModelClass = TemporalFusionAttentionModel if use_attention else TemporalFusionModel
    model = ModelClass(
        embedding_dim=embedding_dim,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_labels = []
    all_predictions = []
    all_fake_probs = []

    with torch.no_grad():
        for sequences, seq_lens, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences, seq_lens)

            probs = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_fake_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc_roc = roc_auc_score(all_labels, all_fake_probs)

    print("\n" + "=" * 60)
    print("CROSS-DATASET TEST RESULTS (trained on FaceForensics++, tested on Celeb-DF)")
    print("=" * 60)
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"AUC-ROC:      {auc_roc:.4f}")
    print("=" * 60)

    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "auc_roc": auc_roc,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset test: FaceForensics++-trained model evaluated on Celeb-DF.")
    parser.add_argument("--real_embeddings_root", required=True)
    parser.add_argument("--fake_embeddings_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lstm_hidden_size", type=int, default=256)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument("--rnn_hidden_size", type=int, default=256)
    parser.add_argument("--rnn_num_layers", type=int, default=1)
    parser.add_argument("--use_attention", action="store_true")
    args = parser.parse_args()

    run_cross_dataset_test(
        real_embeddings_root=args.real_embeddings_root,
        fake_embeddings_root=args.fake_embeddings_root,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        use_attention=args.use_attention,
    )


if __name__ == "__main__":
    main()
