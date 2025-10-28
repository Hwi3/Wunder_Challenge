import os
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import DataPoint, ScorerStepByStep
from models import PredictionModel
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm


device = torch.device("cpu")
EPOCHS = 10

class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=100):
        """
        df: DataFrame with columns [seq_ix, step_in_seq, f1, f2, ..., fN]
        seq_len: number of previous steps to use
        """
        self.seq_len = seq_len

        # Sort to ensure correct order
        df = df.sort_values(by=["seq_ix", "step_in_seq"]).reset_index(drop=True)
        
        # Extract feature columns
        self.feature_cols = [str(i) for i in range(32)]
        print("Feature columns:", len(self.feature_cols))
        # Group by sequence
        grouped = df.groupby("seq_ix")
        self.samples = []

        for _, group in grouped:
            feats = group[self.feature_cols].values  # shape (T, N)
            T = len(feats)
            if T <= seq_len:
                continue
            for i in range(T - seq_len):
                X = feats[i:i + seq_len]
                y = feats[i + seq_len]
                self.samples.append((X, y))
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)



if __name__ == "__main__":
    # Check existence of test file
    train_file = "/workspaces/Wunder_Challenge/submission/datasets/train.csv"
    #test_file = "/workspaces/Wunder_Challenge/submission/datasets/test.csv"
    train_df = pd.read_csv(train_file)
    #test_df = pd.read_csv(test_file)
    train_dataset = TimeSeriesDataset(train_df)
    #test_dataset = TimeSeriesDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = PredictionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.6f}")

    # Load data into scorer
    ##############################

    #TESTING CODE ONLY#
    ##############################
    scorer = ScorerStepByStep(test_file)

    print("Testing simple model with moving average...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    # Evaluate our solution
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nTotal features: {len(scorer.features)}")

    print("\n" + "=" * 60)
    print("Try submitting an archive with solution.py file")
    print("to test the solution submission mechanism!")
    print("=" * 60)
