import pandas as pd
import numpy as np

print("Checking ID distributions...")
train = pd.read_parquet("train.parquet", columns=["ID", "CV_GROUP", "SO3_T", "TARGET"])
test = pd.read_parquet("test.parquet", columns=["ID", "CV_GROUP", "SO3_T"])

print(f"Train IDs: min={train['ID'].min()}, max={train['ID'].max()}")
print(f"Test IDs: min={test['ID'].min()}, max={test['ID'].max()}")

if len(np.intersect1d(train["ID"], test["ID"])) > 0:
    print("WARNING: IDs OVERLAP BETWEEN TRAIN AND TEST!")
else:
    print("Target IDs do not overlap.")

# Plot target vs ID if possible, or calculate correlation
if train["TARGET"].notna().any():
    corr = np.corrcoef(train["ID"], train["TARGET"].fillna(0))[0, 1]
    print(f"Correlation between ID and TARGET: {corr:.6f}")
