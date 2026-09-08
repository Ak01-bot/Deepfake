"""
evaluate_temporal.py

Runs the trained LSTM+RNN fusion model against the held-out TEST set
(never seen during training or validation) and reports the full set
of metrics: accuracy, precision, recall, F1, and AUC-ROC.

This is the number that actually gets reported as the model's real
performance — val_acc during training is useful for tracking progress
and picking the best checkpoint, but the test set is what confirms
the model generalizes to genuinely unseen data.

IMPORTANT: the --lstm_hidden_size / --lstm_num_layers / --rnn_hidden_size
/ --rnn_num_layers flags here must match whatever the checkpoint was
actually trained with (train_temporal.py's flags of the same name),
or loading the checkpoint's weights will fail/mismatch.

Usage:
    python evaluate_temporal.py --embeddings_root embeddings --checkpoint checkpoints/temporal_fusion_best.pth
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from sequence_dataset import VideoSequenceDataset, build_sequence_splits, collate_sequences
from temporal_model import TemporalFusionModel, TemporalFusionAttentionModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_embedding_dim(sample_path) -> int:
    sample = np.load(sample_path)
    return sample.shape[1]


def evaluate(
    embeddings_root: str,
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

    _, _, test_samples = build_sequence_splits(embeddings_root)

    if not test_samples:
        raise ValueError("No test samples found — check embeddings_root path.")

    embedding_dim = infer_embedding_dim(test_samples[0][0])

    test_dataset = VideoSequenceDataset(test_samples)
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
    all_fake_probs = []  # probability of class 1 (fake), needed for AUC-ROC

    with torch.no_grad():
        for sequences, seq_lens, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences, seq_lens)

            probs = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_fake_probs.extend(probs[:, 1].cpu().numpy())  # class 1 = fake

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc_roc = roc_auc_score(all_labels, all_fake_probs)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS (held-out, never seen in training/val)")
    print("=" * 50)
    print(f"Test videos:  {len(test_samples)}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"AUC-ROC:      {auc_roc:.4f}")
    print("=" * 50)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained temporal fusion model on the test set.")
    parser.add_argument("--embeddings_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lstm_hidden_size", type=int, default=256)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument("--rnn_hidden_size", type=int, default=256)
    parser.add_argument("--rnn_num_layers", type=int, default=1)
    parser.add_argument("--use_attention", action="store_true",
                         help="Must match whether the checkpoint was trained with --use_attention.")
    args = parser.parse_args()

    evaluate(
        args.embeddings_root, args.checkpoint, args.batch_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        use_attention=args.use_attention,
    )


if __name__ == "__main__":
    main()
