import pandas as pd
import numpy as np

print("Loading 50k rows to check formula...")
train = pd.read_parquet("train.parquet", columns=["CV_GROUP", "SO3_T", "TARGET"]).head(50000)

train = train.sort_values(["CV_GROUP", "SO3_T"]).reset_index(drop=True)

for shift in range(1, 10):
    fut_target = train.groupby("CV_GROUP")["TARGET"].shift(-shift)
    
    mask = train["TARGET"].notna() & fut_target.notna()
    if mask.sum() > 0:
        r2 = np.corrcoef(train.loc[mask, "TARGET"].values, fut_target[mask].values)[0,1]**2
        print(f"Shift +{shift}: correlation^2 with future target = {r2:.6f}")
        
    past_target = train.groupby("CV_GROUP")["TARGET"].shift(shift)
    mask = train["TARGET"].notna() & past_target.notna()
    if mask.sum() > 0:
        r2 = np.corrcoef(train.loc[mask, "TARGET"].values, past_target[mask].values)[0,1]**2
        print(f"Shift -{shift}: correlation^2 with past target = {r2:.6f}")
