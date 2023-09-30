import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)

class EnhancedPyTorchMLPModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        hidden_dim = kwargs.get("hidden_dim", 256)
        dropout_percent = kwargs.get("dropout_percent", 0.2)
        n_layer = kwargs.get("n_layer", 1)
        rnn_type = kwargs.get("rnn_type", "lstm")
        hidden_rnn_dim = kwargs.get("hidden_rnn_dim", 64)
        num_rnn_layers = kwargs.get("num_rnn_layers", 1)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.Sequential(*[Block(hidden_dim, dropout_percent) for _ in range(n_layer)])
        
        # Add attention mechanism
        self.attention = Attention(hidden_dim)

        # Residual connection between MLP and RNN
        self.residual_connection = nn.Linear(hidden_dim, hidden_rnn_dim)

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(hidden_dim, hidden_rnn_dim, num_layers=num_rnn_layers, batch_first=True)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(hidden_dim, hidden_rnn_dim, num_layers=num_rnn_layers, batch_first=True)

        self.output_layer = nn.Linear(hidden_rnn_dim, output_dim)
        self.bn_output = nn.BatchNorm1d(hidden_rnn_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_percent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_input(self.input_layer(x)))
        x = self.dropout(x)
        x = self.blocks(x)

        # Apply the attention mechanism
        attention_output = self.attention(x)

        # Apply the residual connection
        x = x + self.residual_connection(attention_output)

        # Apply the RNN layer to the output of the MLP
        rnn_out, _ = self.rnn(x)

        x = self.bn_output(rnn_out[:, -1, :])  # Use the last time step's output
        x = self.output_layer(x)
        return x

class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Add attention components (e.g., self-attention mechanism)
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        q = q.view(-1, self.hidden_dim, 1)
        k = k.view(-1, 1, self.hidden_dim)
        v = v.view(-1, self.hidden_dim, 1)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k)
        attn_scores = attn_scores / (self.hidden_dim ** 0.5)
        
        # Apply softmax to obtain attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.view(-1, self.hidden_dim)
        
        # Apply an output linear layer
        output = self.out_linear(attn_output)
        return output

class Block(nn.Module):
    """
    A building block for a multi-layer perceptron (MLP).

    :param hidden_dim: The number of hidden units in the feedforward network.
    :param dropout_percent: The dropout rate for regularization.

    :returns: torch.Tensor. with shape (batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim: int, dropout_percent: int):
        super().__init__()
        self.ff = FeedForward(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_percent)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff(self.ln(x))
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    A simple fully-connected feedforward neural network block.

    :param hidden_dim: The number of hidden units in the block.
    :return: torch.Tensor. with shape (batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
