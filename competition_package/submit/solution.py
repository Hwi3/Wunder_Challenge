import numpy as np
import torch
import torch.nn as nn

class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    #
    state: np.ndarray


class PredictionModel(nn.Module):
    """
    LSTM
    """
    def __init__(self, input_dim=32, hidden_dim=256, num_layers=2, output_dim=32):
        super().__init__()
        self.current_seq_ix = None
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        state_dict = torch.load("weights/lstm_w256_2_E5+DO.pt", map_location='cpu')
        self.load_state_dict(state_dict, strict=False)


    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        x = self.fc(out[:, -1, :])
        return x
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        ## Predict Next State

        # For every new Sequence, reset the history
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.idx_prev100_list = []
    
        # First 100 steps, just store the history
        self.idx_prev100_list.append(data_point.state.copy())
        if not data_point.need_prediction:
            return None

        ## Prediction starts
        # Reset the history to last 100 steps
        if len(self.idx_prev100_list) > 100:
            self.idx_prev100_list.pop(0)
            #self.idx_prev100_list = self.idx_prev100_list[-100:]

            
        # Prepare Input Tensor
        input_tensor = torch.tensor(self.idx_prev100_list, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 100, feature_dim)

        out = self.forward(input_tensor)
        out = out.reshape(32, -1)
        return out.detach().numpy()


class PredictionModel(nn.Module):
    """
    LSTM
    """
    def __init__(self, input_dim=32, hidden_dim=256, num_layers=3, output_dim=32):
        super().__init__()

        self.current_seq_ix = None
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        state_dict = torch.load("weights/lstm_w256_2_E5+DO.pt", map_location='cpu')
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        x = self.fc(out[:, -1, :])
        return x
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        ## Predict Next State

        # For every new Sequence, reset the history
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.idx_prev100_list = []
    
        # First 100 steps, just store the history
        self.idx_prev100_list.append(data_point.state.copy())
        if not data_point.need_prediction:
            return None

        ## Prediction starts
        # Reset the history to last 100 steps
        if len(self.idx_prev100_list) > 100:
            self.idx_prev100_list.pop(0)
            #self.idx_prev100_list = self.idx_prev100_list[-100:]

            
        # Prepare Input Tensor
        input_tensor = torch.tensor(self.idx_prev100_list, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 100, feature_dim)

        out = self.forward(input_tensor)
        out = out.reshape(32, -1)
        return out.detach().numpy()

PredictionModel()
