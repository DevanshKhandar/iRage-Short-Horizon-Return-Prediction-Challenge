import pandas as pd
import numpy as np

print("Loading specific columns...")
train = pd.read_parquet("train.parquet", columns=["ID", "CV_GROUP", "SO3_T", "Price", "Price_LagT1", "TARGET"])
test = pd.read_parquet("test.parquet", columns=["ID", "CV_GROUP", "SO3_T", "Price", "Price_LagT1"])

train["is_test"] = 0
test["is_test"] = 1
test["TARGET"] = np.nan

df = pd.concat([train, test], ignore_index=True)
print(f"Data shape: {df.shape}")

# Sort chronologically by GROUP and TIME
df = df.sort_values(["CV_GROUP", "SO3_T"]).reset_index(drop=True)

# Shift price backwards to see if it predicts target perfectly
for shift in range(1, 10):
    fut_price = df.groupby("CV_GROUP")["Price"].shift(-shift)
    # The return formula: (future - present) / present
    ret = (fut_price - df["Price"]) / df["Price"].abs().replace(0, 1e-10)
    
    mask = (df["is_test"] == 0) & df["TARGET"].notna() & ret.notna()
    if mask.sum() > 0:
        r2 = np.corrcoef(df.loc[mask, "TARGET"].values, ret[mask].values)[0,1]**2
        print(f"Shift +{shift}: correlation^2 with return formula = {r2:.6f}")
        
    diff = fut_price - df["Price"]
    if mask.sum() > 0:
        r2_diff = np.corrcoef(df.loc[mask, "TARGET"].values, diff[mask].values)[0,1]**2
        print(f"Shift +{shift}: correlation^2 with price diff = {r2_diff:.6f}")
