import pandas as pd
import numpy as np

print("Checking ID sort...")
train = pd.read_parquet("train.parquet", columns=["ID", "TARGET"]).head(50000)

train = train.sort_values("ID").reset_index(drop=True)

for shift in range(1, 10):
    fut_target = train["TARGET"].shift(-shift)
    mask = train["TARGET"].notna() & fut_target.notna()
    if mask.sum() > 0:
        r2 = np.corrcoef(train.loc[mask, "TARGET"].values, fut_target[mask].values)[0,1]**2
        print(f"Shift +{shift} by ID sorted: correlation^2 = {r2:.6f}")
