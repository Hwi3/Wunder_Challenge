import os
import sys
import torch.nn as nn
import torch.optim as optim

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/..")


import numpy as np
from utils import DataPoint, ScorerStepByStep
from models import PredictionModel
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
EPOCHS = 1
PATH = "weights/v2.pt"

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
    train_file = r"../datasets/train.csv"
    test_file = r"../datasets/test.csv"
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train_dataset = TimeSeriesDataset(train_df)
    test_dataset = TimeSeriesDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = PredictionModel()
    # move model to device before creating optimizer
    model = model.to(device)
    criterion = nn.MSELoss()
    # start lr 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Linear scheduler: multiply lr from 1.0 -> end_factor over EPOCHS epochs
    # end_factor = 0.0005 / 0.01 = 0.05
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=EPOCHS)
 
    # ensure weights directory exists
    dirpath = os.path.dirname(PATH)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
 
    # TensorBoard writer for live loss visualization
    writer = SummaryWriter(log_dir="runs/exp2")
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            print(inputs.shape, targets.shape)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
 
            # log per-batch loss
            writer.add_scalar("train/loss_batch", loss.item(), global_step)
            # log current learning rate (per-batch)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", current_lr, global_step)
            global_step += 1
 
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float("nan")
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)
        print(f"Epoch {epoch+1}: loss={avg_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6e}")
        # step scheduler once per epoch
        scheduler.step()
 
    writer.flush()
    torch.save(model.state_dict(), PATH)
    writer.close()

    # Load data into scorer
    ##############################

    #TESTING CODE ONLY#
    ##############################
    model = PredictionModel(input_dim=32, hidden_dim=128, output_dim=32)
    # load weights from file (use map_location to be safe)
    print("Model fc layer:", model.fc)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    # ScorerStepByStep expects a path to the dataset file, not a DataFrame
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
