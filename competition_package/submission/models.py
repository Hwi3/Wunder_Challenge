import numpy as np
from utils import DataPoint
import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel

# class PredictionModel:
#     """
#     Simple model that predicts the next value as a moving average
#     of all previous values in the current sequence.
#     """

#     def __init__(self):
#         self.current_seq_ix = None
#         self.sequence_history = []

#     def predict(self, data_point: DataPoint) -> np.ndarray:
#         if self.current_seq_ix != data_point.seq_ix:
#             self.current_seq_ix = data_point.seq_ix
#             self.sequence_history = []

#         self.sequence_history.append(data_point.state.copy())

#         if not data_point.need_prediction:
#             return None

#         return np.mean(self.sequence_history, axis=0)
    


class PredictionModel(nn.Module):
    """
    LSTM
    """
    def __init__(self, input_dim=32, hidden_dim=128, num_layers=2, output_dim=32):
        super().__init__()
        self.current_seq_ix = None
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        last_hidden = out[:, -1, :]  # use last time step
        return self.fc(last_hidden)
    
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        ## Predict Next State
        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            output = self.forward(input_tensor)

        # For every new Sequence, reset the history
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.idx_prev100_list = []
    
        # First 100 steps, just store the history
        if not data_point.need_prediction:
            self.idx_prev100_list.append(data_point.state.copy())
            return None

        ## Prediction starts
        # Reset the history to last 100 steps
        if len(self.idx_prev100_list) > 100:
            self.idx_prev100_list = self.idx_prev100_list[-100:]
            
        # Prepare Input Tensor
        input_tensor = torch.tensor(self.idx_prev100_list, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 100, feature_dim)
        input_tensor = input_tensor.to(device)
        return self.forward(input_tensor)
    


# class PredictionModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128):
#         self.sequence_history = []
#         super().__init__()
#         hidden_dim = 128
#         self.input_proj = nn.Linear(input_dim, hidden_dim)

#         config = MambaConfig(
#             hidden_size=hidden_dim,
#             num_hidden_layers=4,
#             intermediate_size=hidden_dim * 2,
#             vocab_size=1,  # unused
#         )
#         self.backbone = MambaModel(config)
#         self.fc = nn.Linear(hidden_dim, input_dim)

#     def forward(self, x):
#         x = self.input_proj(x)  
#         out = self.backbone(inputs_embeds=x).last_hidden_state
#         return self.fc(out)
    
#     def predict(self, data_point: DataPoint) -> np.ndarray:
#         ## Predict Next State
#         if self.current_seq_ix != data_point.seq_ix:
#             self.current_seq_ix = data_point.seq_ix
#             self.sequence_history = []

#         if not data_point.need_prediction:
#             self.sequence_history.append(data_point.state.copy())
#             return None

#         ## Prediction starts

        
#         return self.forward(data_point.state)
