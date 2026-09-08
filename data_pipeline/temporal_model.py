"""
temporal_model.py

Two model variants:

1. TemporalFusionModel — the original architecture from the project
   doc: LSTM + parallel RNN, each branch summarized by its FINAL
   hidden state, concatenated and fused.

2. TemporalFusionAttentionModel — an enhancement (not in the project
   doc's original spec, tested as an optional improvement): instead
   of only using the final hidden state, an attention layer looks at
   EVERY timestep's output and learns to weight which frames mattered
   most for the decision, then combines them into a weighted context
   vector. This can help because a deepfake's most telling artifacts
   might appear in the middle of a sequence, not necessarily be best
   summarized by whatever the LSTM/RNN happened to be "remembering"
   at the very last frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalFusionModel(nn.Module):
    """
    Dual-branch temporal classifier over sequences of CNN frame embeddings.
    Uses each branch's final hidden state as its summary — the
    architecture as originally specified in the project document.
    """

    def __init__(
        self,
        embedding_dim: int,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        dropout: float = 0.4,
        num_classes: int = 2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )

        fusion_input_dim = lstm_hidden_size + rnn_hidden_size
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, padded_sequences: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            padded_sequences, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (lstm_hidden, _) = self.lstm(packed)
        lstm_final = lstm_hidden[-1]

        _, rnn_hidden = self.rnn(packed)
        rnn_final = rnn_hidden[-1]

        fused = torch.cat([lstm_final, rnn_final], dim=1)
        logits = self.fusion_head(fused)
        return logits


class AttentionPooling(nn.Module):
    """
    Learns a weight for each timestep in a sequence, then computes a
    weighted sum ("context vector") instead of just taking the final
    timestep. Padded positions are masked out so they get zero weight
    regardless of what the (arbitrary, padding-value) LSTM/RNN output
    for them looked like.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_output: (batch, max_seq_len, hidden_size) — every timestep's output.
            seq_lens: (batch,) — true length of each sequence, for masking padding.

        Returns:
            (batch, hidden_size) — weighted context vector per sequence.
        """
        batch_size, max_seq_len, _ = sequence_output.shape

        # Raw attention scores per timestep
        scores = self.attention_weights(sequence_output).squeeze(-1)  # (batch, max_seq_len)

        # Mask out padded positions (set their score to -inf so softmax gives them ~0 weight)
        mask = torch.arange(max_seq_len, device=sequence_output.device).unsqueeze(0) < seq_lens.unsqueeze(1).to(sequence_output.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=1)  # (batch, max_seq_len)

        # Weighted sum over the sequence
        context = torch.bmm(attention_weights.unsqueeze(1), sequence_output).squeeze(1)  # (batch, hidden_size)
        return context


class TemporalFusionAttentionModel(nn.Module):
    """
    Same dual-branch structure as TemporalFusionModel, but each branch
    is summarized via attention pooling over ALL timesteps instead of
    just the final hidden state.
    """

    def __init__(
        self,
        embedding_dim: int,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        dropout: float = 0.4,
        num_classes: int = 2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.lstm_attention = AttentionPooling(lstm_hidden_size)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.rnn_attention = AttentionPooling(rnn_hidden_size)

        fusion_input_dim = lstm_hidden_size + rnn_hidden_size
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, padded_sequences: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            padded_sequences, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        # --- LSTM branch: get EVERY timestep's output, not just the final one ---
        lstm_packed_out, _ = self.lstm(packed)
        lstm_sequence_output, _ = pad_packed_sequence(lstm_packed_out, batch_first=True)
        lstm_context = self.lstm_attention(lstm_sequence_output, seq_lens)

        # --- RNN branch: same pattern ---
        rnn_packed_out, _ = self.rnn(packed)
        rnn_sequence_output, _ = pad_packed_sequence(rnn_packed_out, batch_first=True)
        rnn_context = self.rnn_attention(rnn_sequence_output, seq_lens)

        # --- Fusion ---
        fused = torch.cat([lstm_context, rnn_context], dim=1)
        logits = self.fusion_head(fused)
        return logits


if __name__ == "__main__":
    # Sanity check both variants on dummy data before wiring into training.
    batch_size = 4
    max_seq_len = 20
    embedding_dim = 2048

    dummy_sequences = torch.randn(batch_size, max_seq_len, embedding_dim)
    dummy_lens = torch.tensor([20, 15, 10, 5])

    print("Testing TemporalFusionModel (final hidden state)...")
    model = TemporalFusionModel(embedding_dim=embedding_dim)
    output = model(dummy_sequences, dummy_lens)
    assert output.shape == (batch_size, 2)
    print(f"  Output shape: {tuple(output.shape)} - OK")

    print("Testing TemporalFusionAttentionModel (attention pooling)...")
    attention_model = TemporalFusionAttentionModel(embedding_dim=embedding_dim)
    attention_output = attention_model(dummy_sequences, dummy_lens)
    assert attention_output.shape == (batch_size, 2)
    print(f"  Output shape: {tuple(attention_output.shape)} - OK")

    print("\nBoth sanity checks passed.")
