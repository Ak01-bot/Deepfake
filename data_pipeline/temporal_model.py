"""
temporal_model.py

The dual-branch temporal architecture: cached CNN frame embeddings are
fed into a stacked LSTM branch and a parallel RNN branch simultaneously,
and their outputs are fused into a final real/fake classification.

This is the architecture described in the project's Abstract:
"descriptors are then fed into a stacked LSTM network which models
temporal dependencies... Additionally, another stream of input data
passes through a Recurrent Neural Network (RNN) module... Both branches
come together in a classification head."

Architecture diagram:

    Embedding sequence (seq_len, embedding_dim)
              |
    ┌─────────┴─────────┐
    |                    |
  Stacked LSTM      Parallel RNN
  (2 layers)         (1 layer)
    |                    |
  final hidden       final hidden
  state (256-d)       state (128-d)
    |                    |
    └─────────┬─────────┘
          concatenate (384-d)
              |
         Fusion FC layers
              |
        2-class output (real/fake)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalFusionModel(nn.Module):
    """
    Dual-branch temporal classifier over sequences of CNN frame embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        # Stacked LSTM branch — models longer-range temporal dependencies
        # across the frame sequence (the "stacked" refers to
        # lstm_num_layers > 1, per the project doc's "stacked LSTM network").
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # Parallel RNN branch — a second, simpler temporal stream over
        # the same input sequence, per the project doc's "another
        # stream of input data passes through a Recurrent Neural
        # Network (RNN) module, adding temporal context."
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )

        # Fusion head: combines both branches' final hidden states
        fusion_input_dim = lstm_hidden_size + rnn_hidden_size
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, padded_sequences: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            padded_sequences: (batch, max_seq_len, embedding_dim) — padded batch of embedding sequences.
            seq_lens: (batch,) — true (pre-padding) length of each sequence, so the LSTM/RNN
                       ignore the padded positions rather than treating them as real frames.

        Returns:
            (batch, num_classes) logits.
        """
        # pack_padded_sequence tells PyTorch's LSTM/RNN to skip
        # computation on the padded positions, both for correctness
        # (padding shouldn't influence the result) and efficiency.
        packed = pack_padded_sequence(
            padded_sequences, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        # --- LSTM branch ---
        _, (lstm_hidden, _) = self.lstm(packed)
        # lstm_hidden shape: (num_layers, batch, hidden_size) — take the last layer's final state
        lstm_final = lstm_hidden[-1]  # (batch, lstm_hidden_size)

        # --- Parallel RNN branch ---
        _, rnn_hidden = self.rnn(packed)
        rnn_final = rnn_hidden[-1]  # (batch, rnn_hidden_size)

        # --- Fusion ---
        fused = torch.cat([lstm_final, rnn_final], dim=1)  # (batch, lstm_hidden_size + rnn_hidden_size)
        logits = self.fusion_head(fused)
        return logits


if __name__ == "__main__":
    # Quick sanity check: confirm the model runs on dummy data with
    # the expected shapes, before wiring it into real training.
    batch_size = 4
    max_seq_len = 20
    embedding_dim = 2048  # Xception's typical pooled feature size

    dummy_sequences = torch.randn(batch_size, max_seq_len, embedding_dim)
    dummy_lens = torch.tensor([20, 15, 10, 5])  # varying lengths, as in real data

    model = TemporalFusionModel(embedding_dim=embedding_dim)
    output = model(dummy_sequences, dummy_lens)

    print(f"Output shape: {tuple(output.shape)} (expected: ({batch_size}, 2))")
    assert output.shape == (batch_size, 2), "Output shape mismatch!"
    print("Sanity check passed.")
