import numpy as np
from utils import DataPoint
import torch
import torch.nn as nn

##LSTM1 lstm + fc
##LSTM2 lstm + fc1 + fc2
##LSTM3 lstm + fc1 + fc2
##LSTM4 lstm + fc1 + wb
class PredictionModel(nn.Module):
    """
    LSTM
    """
    def __init__(self, input_dim=32, hidden_dim=256, num_layers=1, output_dim=32):
        super().__init__()
        self.current_seq_ix = None
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.weight = nn.Parameter(torch.ones(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out1, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        out2 = self.fc1(out1[:, -1, :])
        out3 = out2 * self.weight + self.bias
        return out3
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        ## Reset for new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            
            # Preallocate tensor buffer once (100, 32)
            self.buffer = torch.zeros(100, 32, dtype=torch.float32)
            self.buffer_count = 0   # how many steps filled so far

        # Add new state into buffer
        new_state = torch.from_numpy(data_point.state).float()  # shape (32,)

        if self.buffer_count < 100:
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
        if self.buffer_count < 100:
            return None

        # Prepare Input Tensor (NO COPY)
        input_tensor = self.buffer.unsqueeze(0)  # shape (1, 100, 32)

        out = self.forward(input_tensor)
        return out.reshape(32, -1).detach().numpy()

    
# class PredictionModel(nn.Module):
#     """
#     LSTM
#     """
#     def __init__(self, input_dim=32, hidden_dim=256, num_layers=2, output_dim=32):
#         super().__init__()
#         self.current_seq_ix = None
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
#         self.fc = nn.Linear(hidden_dim, output_dim)


#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
#         x = self.fc(out[:, -1, :])
#         return x
    
#     def predict(self, data_point: DataPoint) -> np.ndarray:
#         ## Predict Next State

#         # For every new Sequence, reset the history
#         if self.current_seq_ix != data_point.seq_ix:
#             self.current_seq_ix = data_point.seq_ix
#             self.idx_prev100_list = []
    
#         # First 100 steps, just store the history
#         self.idx_prev100_list.append(data_point.state.copy())
#         if not data_point.need_prediction:
#             return None

#         ## Prediction starts
#         # Reset the history to last 100 steps
#         if len(self.idx_prev100_list) > 100:
#             self.idx_prev100_list.pop(0)
#             #self.idx_prev100_list = self.idx_prev100_list[-100:]

            
#         # Prepare Input Tensor
#         input_tensor = torch.tensor(self.idx_prev100_list, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 100, feature_dim)

#         out = self.forward(input_tensor)
#         out = out.reshape(32, -1)
#         return out.detach().numpy()




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
