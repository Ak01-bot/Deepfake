"""
generalization_test.py

The Week 7 generalization test: evaluates the trained model against
the FaceShifter manipulation technique, which was held out from ALL
training/validation/testing throughout the entire project — this
answers the project document's core question (Problem Statement #5):
does the model generalize to a manipulation technique it has never
seen, or does it only work on techniques it was explicitly trained on?

Test set construction:
    - ALL faceshifter_holdout videos (label = fake) — the unseen technique.
    - The REAL test videos from the normal 5-category split (label = real)
      — reused from the original training run's held-out test set, so
      these are real videos the model has also never trained on.

Combining both is what makes the metrics meaningful: evaluating on an
all-fake set alone can't produce a real precision/recall/F1/AUC (those
need both classes present). This mirrors how the model would actually
be used — facing a mix of real and (possibly novel) fake content.

Usage:
    python generalization_test.py --normal_embeddings_root embeddings_verify \\
        --holdout_embeddings_root embeddings_holdout \\
        --checkpoint checkpoints/temporal_attention.pth --use_attention
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


def load_holdout_samples(holdout_embeddings_root: str) -> list:
    """Every video in the holdout set is fake (label 1) — no split needed, use all of them as test data."""
    from pathlib import Path
    holdout_dir = Path(holdout_embeddings_root)
    video_files = sorted(holdout_dir.glob("*.npy"))
    return [(f, 1) for f in video_files]


def run_generalization_test(
    normal_embeddings_root: str,
    holdout_embeddings_root: str,
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

    # Get the real test videos from the normal (5-category) split — same
    # seed/logic as everything else, so this is the exact same held-out
    # real test set used throughout the project, not a new random sample.
    _, _, normal_test_samples = build_sequence_splits(normal_embeddings_root)
    real_test_samples = [(path, label) for path, label in normal_test_samples if label == 0]
    print(f"\nReal test videos (from normal split): {len(real_test_samples)}")

    # Get ALL faceshifter_holdout videos — every single one, all labeled fake.
    holdout_samples = load_holdout_samples(holdout_embeddings_root)
    print(f"Holdout fake videos (FaceShifter, unseen technique): {len(holdout_samples)}")

    combined_samples = real_test_samples + holdout_samples
    print(f"Combined generalization test set: {len(combined_samples)} videos\n")

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

    # Track holdout-only accuracy separately too — this is the number
    # that most directly answers "does it generalize to the unseen
    # technique", distinct from the combined accuracy which also
    # includes the (already-seen-category) real videos.
    holdout_correct = 0
    holdout_total = 0

    with torch.no_grad():
        sample_index = 0
        for sequences, seq_lens, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences, seq_lens)

            probs = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            batch_size_actual = labels.size(0)
            for i in range(batch_size_actual):
                sample_path, _ = combined_samples[sample_index]
                is_holdout = sample_index >= len(real_test_samples)
                if is_holdout:
                    holdout_total += 1
                    if predictions[i].item() == labels[i].item():
                        holdout_correct += 1
                sample_index += 1

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_fake_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc_roc = roc_auc_score(all_labels, all_fake_probs)
    holdout_accuracy = holdout_correct / holdout_total if holdout_total > 0 else 0.0

    print("\n" + "=" * 60)
    print("WEEK 7 GENERALIZATION TEST RESULTS")
    print("(FaceShifter — a manipulation technique NEVER used in training)")
    print("=" * 60)
    print(f"Combined test set:        {len(combined_samples)} videos "
          f"({len(real_test_samples)} real + {len(holdout_samples)} unseen-fake)")
    print(f"\n--- Combined metrics (real + unseen-fake together) ---")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"AUC-ROC:      {auc_roc:.4f}")
    print(f"\n--- Holdout-only accuracy (the direct generalization number) ---")
    print(f"Correctly identified as fake: {holdout_correct}/{holdout_total} "
          f"({holdout_accuracy:.4f})")
    print("=" * 60)

    return {
        "combined_accuracy": accuracy,
        "combined_precision": precision,
        "combined_recall": recall,
        "combined_f1": f1,
        "combined_auc_roc": auc_roc,
        "holdout_only_accuracy": holdout_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Week 7 generalization test on the held-out FaceShifter technique.")
    parser.add_argument("--normal_embeddings_root", required=True,
                         help="Path to the regular 5-category embeddings (to get the real test videos).")
    parser.add_argument("--holdout_embeddings_root", required=True,
                         help="Path to the faceshifter_holdout embeddings (extract_embeddings_holdout.py output).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lstm_hidden_size", type=int, default=256)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument("--rnn_hidden_size", type=int, default=256)
    parser.add_argument("--rnn_num_layers", type=int, default=1)
    parser.add_argument("--use_attention", action="store_true",
                         help="Must match how the checkpoint was actually trained.")
    args = parser.parse_args()

    run_generalization_test(
        normal_embeddings_root=args.normal_embeddings_root,
        holdout_embeddings_root=args.holdout_embeddings_root,
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
