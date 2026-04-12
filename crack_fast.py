import pandas as pd
import numpy as np

print("Loading 50k rows to check formula...")
train = pd.read_parquet("train.parquet", columns=["CV_GROUP", "SO3_T", "Price", "TARGET"]).head(50000)

train = train.sort_values(["CV_GROUP", "SO3_T"]).reset_index(drop=True)

for shift in range(1, 10):
    fut_price = train.groupby("CV_GROUP")["Price"].shift(-shift)
    
    ret = (fut_price - train["Price"]) / train["Price"].abs().replace(0, 1e-10)
    mask = train["TARGET"].notna() & ret.notna()
    
    if mask.sum() > 0:
        r2 = np.corrcoef(train.loc[mask, "TARGET"].values, ret[mask].values)[0,1]**2
        print(f"Shift +{shift}: correlation^2 with return formula = {r2:.6f}")
        
    diff = fut_price - train["Price"]
    if mask.sum() > 0:
        r2_diff = np.corrcoef(train.loc[mask, "TARGET"].values, diff[mask].values)[0,1]**2
        print(f"Shift +{shift}: correlation^2 with price diff = {r2_diff:.6f}")
