import numpy as np
from utils import DataPoint
import torch
import torch.nn as nn

##LSTM1 lstm + fc
##LSTM2 lstm + fc1 + fc2
##LSTM3 lstm + fc1 + fc2
##LSTM4 lstm + fc1 + wb
##LSTM5 double LSTM 
##LSTM7 double LSTM 30, 10
##LSTREE 


class PredictionModel(nn.Module):
    """
    LSTM
    """
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2, output_dim=32):
        super().__init__()
        self.n_back = 100
        self.current_seq_ix = None
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.tree = SoftDecisionTree(in_features = hidden_dim, depth = 3, out_dim=output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out1, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        h_last = out1[:, -1, :]
        out2 = self.tree(h_last) 
        return out2
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        ## Reset for new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            
            # Preallocate tensor buffer once (100, 32)
            self.buffer = torch.zeros(self.n_back, 32, dtype=torch.float32)
            self.buffer_count = 0   # how many steps filled so far

        # Add new state into buffer
        new_state = torch.from_numpy(data_point.state).float()  # shape (32,)

        if self.buffer_count < self.n_back:
            # still filling initial window
            self.buffer[self.buffer_count] = new_state
            self.buffer_count += 1
        else:
            # sliding the window
            self.buffer[:-1] = self.buffer[1:].clone()
            self.buffer[-1] = new_state

        # If not time to predict
        if not data_point.need_prediction:
            return None
        
        # Need at least 100 steps to produce prediction
        if self.buffer_count < self.n_back:
            return None

        # Prepare Input Tensor (NO COPY)
        input_tensor = self.buffer.unsqueeze(0)  # shape (1, 100, 32)

        out = self.forward(input_tensor)
        return out.reshape(32, -1).detach().numpy()

class SoftDecisionTree(nn.Module):
    """
    Differentiable binary decision tree.
    depth=3 creates 8 leaves and 7 internal nodes.
    The output_dim matches your 32-dim target.
    """
    def __init__(self, in_features, depth=3, out_dim=32):
        super().__init__()
        self.in_features = in_features
        self.depth = depth
        self.out_dim = out_dim

        self.num_internal_nodes = 2 ** depth - 1
        self.num_leaves = 2 ** depth

        # gating: linear layer for each internal node
        self.node_weights = nn.Parameter(
            torch.randn(self.num_internal_nodes, in_features) * 0.1
        )
        self.node_bias = nn.Parameter(torch.zeros(self.num_internal_nodes))

        # outputs: one vector per leaf
        self.leaf_values = nn.Parameter(
            torch.randn(self.num_leaves, out_dim) * 0.1
        )

    def forward(self, x):
        """
        x: (B, in_features)
        returns: (B, out_dim)
        """
        B = x.size(0)

        # gate probabilities for internal nodes
        logits = x @ self.node_weights.t() + self.node_bias  # (B, num_internal)
        gates = torch.sigmoid(logits)

        # propagate down tree
        path_probs = x.new_ones(B, 1)

        for level in range(self.depth):
            start = 2 ** level - 1
            end = 2 ** (level + 1) - 1
            g = gates[:, start:end]  # (B, 2^level)

            left = path_probs * (1 - g)
            right = path_probs * g
            path_probs = torch.cat([left, right], dim=1)

        # weighted sum of leaf outputs
        out = path_probs @ self.leaf_values  # (B, out_dim)
        return out
