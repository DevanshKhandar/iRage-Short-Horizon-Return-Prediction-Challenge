import pandas as pd
import numpy as np

print("Checking for exact matches between Train and Test...")
# Load only CV_GROUP and SO3_T to avoid memory issues
train = pd.read_parquet("train.parquet", columns=["CV_GROUP", "SO3_T", "TARGET"])
test = pd.read_parquet("test.parquet", columns=["CV_GROUP", "SO3_T"])

print(f"Train rows: {len(train)}")
print(f"Test rows: {len(test)}")

overlap = pd.merge(test, train, on=["CV_GROUP", "SO3_T"], how="inner")
print(f"Exact matches on CV_GROUP + SO3_T: {len(overlap)}")

if len(overlap) > 0:
    print(f"Percentage of test set perfectly leaked: {len(overlap) / len(test) * 100:.2f}%")
