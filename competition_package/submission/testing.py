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
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
EPOCHS = 5
#PATH = r"/workspaces/Wunder_Challenge/competition_package/submission/weights/v3.pt"
MODEL = "lstm2_w256_2_E5+DO"
PATH = f"weights/{MODEL}.pt"

training = True

class TimeSeriesDataset(Dataset):
    def __init__(self, df, n_back=100):
        grouped = df.groupby("seq_ix")
        self.all_data = []
        for _, group in tqdm(grouped):
            group = group.drop(columns=['seq_ix','step_in_seq', 'need_prediction'])
            for time_step in range(n_back-1,999):
                window = group.iloc[time_step - n_back: time_step + 1]
                if not window.empty:
                    self.all_data.append(group.iloc[time_step-n_back:time_step+1])

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        dat = torch.tensor(self.all_data[idx].to_numpy(), dtype=torch.float32)
        X, y = dat[:100], dat[100]
        return X, y



if __name__ == "__main__":

    # Check existence of test file
    train_file = r"C:\Users\hwisa\OneDrive\문서\Projects\Wunder_Challenge\competition_package\datasets\train.parquet"
    #train_file = "/workspaces/Wunder_Challenge/competition_package/datasets/train.parquet"
    train_df = pd.read_parquet(train_file)
    train_dataset = TimeSeriesDataset(train_df, n_back=100)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = PredictionModel()
    # move model to device before creating optimizer
    model = model.to(device)
    criterion = nn.MSELoss()
    # start lr 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Linear scheduler: multiply lr from 1.0 -> end_factor over EPOCHS epochs
    # end_factor = 0.0005 / 0.01 = 0.05
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1e-3, total_iters=EPOCHS)
 
    # ensure weights directory exists
    dirpath = os.path.dirname(PATH)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
 
    # TensorBoard writer for live loss visualization
    writer = SummaryWriter(log_dir=f"runs/{MODEL}")
    global_step = 0

    if training:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                # break
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
    model = PredictionModel()
    # load weights from file (use map_location to be safe)
    print("Model fc layer:", model.fc)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    # ScorerStepByStep expects a path to the dataset file, not a DataFrame
    #test = "/workspaces/Wunder_Challenge/competition_package/datasets/test.csv"
    test = r"C:\Users\hwisa\OneDrive\문서\Projects\Wunder_Challenge\competition_package\datasets\test.csv"
    scorer = ScorerStepByStep(test,MODEL)

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
