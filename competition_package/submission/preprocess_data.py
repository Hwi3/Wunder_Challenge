import pandas as pd
import numpy as np
from tqdm.auto import tqdm


SPLIT = 0.8
all_file = r"C:\Users\hwisa\OneDrive\문서\Projects\Wunder_Challenge\competition_package\datasets\train.parquet"
df = pd.read_parquet(all_file)

grouped = df.groupby("seq_ix")
n_back = 100
train = []
test = []

for data_idx, group in tqdm(grouped):
    group = group.drop(columns=['seq_ix','step_in_seq', 'need_prediction'])
    for time_step in range(n_back-1,999):
        window = group.iloc[time_step - n_back: time_step + 1]
        if not window.empty:
            train.append(window)

train = np.array(train)
print()
train, test = np.split(train, [int(SPLIT * len(train))])
np.save(r"C:\Users\hwisa\OneDrive\문서\Projects\Wunder_Challenge\competition_package\datasets\train.npy", train)
np.save(r"C:\Users\hwisa\OneDrive\문서\Projects\Wunder_Challenge\competition_package\datasets\test.npy", test)



    # feats = group[self.feature_cols].values.astype(float)  # shape (T, N)
    # # apply standardization using train stats (or provided stats)
    # feats = (feats - self.mean) / (self.std + self.eps)
    # T = len(feats)
    # if T <= seq_len:
    #     continue
    # for i in range(T - seq_len):
    #     X = feats[i:i + seq_len]
    #     y = feats[i + seq_len]
    #     self.samples.append((X, y))