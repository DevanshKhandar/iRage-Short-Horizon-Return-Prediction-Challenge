import pandas as pd
import numpy as np

print("Loading data to check for exact row duplicates...")
# Just check a few highly specific columns to see if there are exact feature duplicates
cols = ["S01_F01_U01", "Price", "S01_O02", "S02_F01_U01"]
train = pd.read_parquet("train.parquet", columns=cols + ["TARGET"])
test = pd.read_parquet("test.parquet", columns=cols + ["ID"])

print(f"Train: {len(train)}, Test: {len(test)}")

overlap = pd.merge(test, train, on=cols, how="inner")
print(f"Found {len(overlap)} matching rows across train and test!")

if len(overlap) > 0:
    print(f"Percentage of test set duplicated: {len(overlap) / len(test) * 100:.2f}%")
