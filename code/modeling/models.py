import torch
import torch.nn as nn


class StateCentricLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()

        # Input: State Embedding (D) + Goal Embedding (D)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # We concatenate State + Goal at every step
        self.lstm = nn.LSTM(
            input_size=input_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Project back to State Embedding dimension
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, state_seq, goal_seq, lengths=None, hidden=None):
        """
        state_seq: [B, T, D]
        goal_seq: [B, D] (will be expanded to [B, T, D])
        """
        B, T, D = state_seq.shape

        # Expand goal to match sequence length
        goal_expanded = goal_seq.unsqueeze(1).expand(-1, T, -1)

        # Concatenate: [B, T, 2D]
        lstm_input = torch.cat([state_seq, goal_expanded], dim=2)

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
