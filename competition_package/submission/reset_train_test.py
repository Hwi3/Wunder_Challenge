import os
import pandas as pd
import numpy as np

RATIO = 0.8
test_file = "../datasets/train.parquet"

df = pd.read_parquet(test_file)

unique_sequences = df["seq_ix"].unique()
np.random.seed(1)
np.random.shuffle(unique_sequences)

train_size = int(RATIO * len(unique_sequences))
train_seq_ixs = set(df['seq_ix'].unique()[:train_size])
test_seq_ixs = set(df['seq_ix'].unique()[train_size:])
train_df = df[df['seq_ix'].isin(train_seq_ixs)]
test_df = df[df['seq_ix'].isin(test_seq_ixs)]
train_df.to_csv("../datasets/train.csv", index=False)
test_df.to_csv("../datasets/test.csv", index=False)