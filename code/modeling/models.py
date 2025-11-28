import torch
import torch.nn as nn


class StateCentricLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, embed_dim=32):
        super().__init__()

        # 1. Add a Projection Layer (Dimensionality Reduction)
        self.encoder = nn.Linear(input_dim, embed_dim)

        # LSTM takes the smaller embed_dim
        # Input: State (embed_dim) + Goal (embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Head projects back to FULL input_dim (to match WL targets)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, state_seq, goal_seq, lengths=None, hidden=None):
        """
        state_seq: [B, T, Input_D]
        goal_seq: [B, D] (will be expanded to [B, T, D])
        """

        # 1. Project High-Dim Sparse -> Low-Dim Dense
        state_emb = torch.relu(self.encoder(state_seq))  # [B, T, embed_dim]

        # Project Goal too (using same encoder to align spaces)
        goal_emb = torch.relu(self.encoder(goal_seq))  # [B, embed_dim]

        _, T, _ = state_emb.shape

        # Expand goal to match sequence length
        goal_expanded = goal_emb.unsqueeze(1).expand(-1, T, -1)

        # Concatenate: [B, T, 2D]
        lstm_input = torch.cat([state_emb, goal_expanded], dim=2)

        # Pack sequences if lengths are provided
        if lengths is not None:
            lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
                lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM Forward
        # out: [B, T, H]
        out, hidden = self.lstm(lstm_input, hidden)

        if lengths is not None:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Project to next state
        pred_next_state = self.head(out)

        return pred_next_state, hidden
