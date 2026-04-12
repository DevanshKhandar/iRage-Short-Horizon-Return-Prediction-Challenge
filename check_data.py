"""Quick data quality check."""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc

# Check target and metadata
print("=== TARGET & METADATA ===")
t = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
target = t["TARGET"]
print(f"NaN in TARGET: {target.isna().sum()}")
print(f"Inf in TARGET: {np.isinf(target).sum()}")
print(f"NaN in ID: {t['ID'].isna().sum()}")
print(f"NaN in CV_GROUP: {t['CV_GROUP'].isna().sum()}")
print(f"CV_GROUP unique: {t['CV_GROUP'].nunique()}")
print(f"CV_GROUP values: {sorted(t['CV_GROUP'].unique())}")
print(f"TARGET stats: mean={target.mean():.6f}, std={target.std():.6f}")
del t; gc.collect()

# Check features
print("\n=== FEATURE QUALITY ===")
pf = pq.ParquetFile("train.parquet")
cols = [c for c in pf.schema.names if c not in {"ID", "CV_GROUP", "TARGET"}]
print(f"Total features: {len(cols)}")

nan_feats = []
inf_feats = []
const_feats = []

for i in range(0, len(cols), 50):
    batch = cols[i:i+50]
    df = pd.read_parquet("train.parquet", columns=batch)
    for c in batch:
        nc = df[c].isna().sum()
        ic = np.isinf(df[c].astype(float)).sum()
        sc = df[c].std()
        if nc > 0:
            nan_feats.append((c, int(nc)))
        if ic > 0:
            inf_feats.append((c, int(ic)))
        if sc == 0:
            const_feats.append(c)
    del df
    gc.collect()
    if (i + 50) % 100 == 0:
        print(f"  Checked {min(i+50, len(cols))}/{len(cols)}")

print(f"\nFeatures with NaN: {len(nan_feats)}")
if nan_feats:
    for c, n in nan_feats[:10]:
        print(f"  {c}: {n} NaN")

print(f"Features with Inf: {len(inf_feats)}")
if inf_feats:
    for c, n in inf_feats[:10]:
        print(f"  {c}: {n} Inf")

print(f"Constant features (std=0): {len(const_feats)}")
if const_feats:
    for c in const_feats[:10]:
        print(f"  {c}")

# Check test data too
print("\n=== TEST DATA ===")
pf_test = pq.ParquetFile("test.parquet")
test_cols = pf_test.schema.names
test_feats = [c for c in test_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
train_only = [c for c in cols if c not in test_cols]
test_only = [c for c in test_feats if c not in cols]
print(f"Test features: {len(test_feats)}")
print(f"In train but not test: {len(train_only)}")
if train_only:
    print(f"  {train_only[:5]}")
print(f"In test but not train: {len(test_only)}")

print("\nData quality check complete!")
