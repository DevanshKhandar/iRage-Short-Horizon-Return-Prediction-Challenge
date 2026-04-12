"""Deep investigation of the data to find the hidden signal."""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Load key columns
print("Loading data...")
cols_to_load = [
    "TARGET", "CV_GROUP", "ID",
    "Price", "Price_LagT1", "Price_LagT2", "Price_LagT3",
    "S01_O02", "S01_O02_LagT1", "S01_O02_A01",
    "S02_O02_A01_LagT3", "S02_O02_LagT3",
    "SO3_T",
    "S01_F01_U01", "S01_F02_U01", "S01_F03_U01",
    "S02_F01_U01", "S02_F02_U01", "S02_F03_U01",
    "S03_V06_V01", "S03_V07_V06",
]
train = pd.read_parquet("train.parquet", columns=cols_to_load)
y = train["TARGET"].values

print(f"\n=== TARGET stats ===")
print(f"  mean: {y.mean():.8f}")
print(f"  std:  {y.std():.8f}")
print(f"  min:  {y.min():.6f}")
print(f"  max:  {y.max():.6f}")

print(f"\n=== CV_GROUP distribution ===")
print(train["CV_GROUP"].value_counts().sort_index())

print(f"\n=== SO3_T (possible timestamp) ===")
print(f"  nunique: {train['SO3_T'].nunique()}")
print(f"  min: {train['SO3_T'].min()}")
print(f"  max: {train['SO3_T'].max()}")
print(f"  sample: {train['SO3_T'].head(10).values}")

print(f"\n=== Price stats ===")
print(f"  mean: {train['Price'].mean():.4f}")
print(f"  std:  {train['Price'].std():.4f}")
print(f"  range: [{train['Price'].min():.4f}, {train['Price'].max():.4f}]")

# ==============================================================================
# TEST HYPOTHESIS: TARGET = some function of Price columns
# ==============================================================================
print("\n=== Testing Price-based hypotheses ===")

# 1. Percentage return: (Price - Price_LagT1) / Price_LagT1
pct_ret = (train["Price"] - train["Price_LagT1"]) / train["Price_LagT1"]
mask = np.isfinite(pct_ret)
r2 = r2_score(y[mask], pct_ret[mask])
corr = np.corrcoef(y[mask], pct_ret[mask])[0, 1]
print(f"  (Price - Price_LagT1) / Price_LagT1:  R2={r2:.6f}, corr={corr:.6f}")

# 2. Simple difference
diff = train["Price"] - train["Price_LagT1"]
r2 = r2_score(y, diff)
corr = np.corrcoef(y, diff)[0, 1]
print(f"  Price - Price_LagT1:                   R2={r2:.6f}, corr={corr:.6f}")

# 3. Ratio
ratio = train["Price"] / train["Price_LagT1"]
mask = np.isfinite(ratio)
r2 = r2_score(y[mask], ratio[mask])
print(f"  Price / Price_LagT1:                   R2={r2:.6f}")

# 4. Log return
log_ret = np.log(train["Price"] / train["Price_LagT1"])
mask = np.isfinite(log_ret)
r2 = r2_score(y[mask], log_ret[mask])
print(f"  log(Price / Price_LagT1):              R2={r2:.6f}")

# ==============================================================================
# TEST: Engineered features with higher-order interactions
# ==============================================================================
print("\n=== Testing feature combinations ===")

# Bid-ask spread features
for f1 in ["S01_F01_U01", "S01_F02_U01", "S01_F03_U01"]:
    for f2 in ["S02_F01_U01", "S02_F02_U01", "S02_F03_U01"]:
        diff = train[f1] - train[f2]
        corr = np.corrcoef(y, diff)[0, 1]
        if abs(corr) > 0.05:
            print(f"  {f1} - {f2}: corr={corr:.6f}")

# S01_O02 type features (highest individual correlations)
for c in ["S01_O02", "S01_O02_LagT1", "S01_O02_A01"]:
    corr = np.corrcoef(y, train[c])[0, 1]
    print(f"  {c}: corr={corr:.6f}")

# Check order book imbalance type features
oi = train["S01_O02"] - train["S01_O02_A01"]
corr = np.corrcoef(y, oi)[0, 1]
print(f"  S01_O02 - S01_O02_A01: corr={corr:.6f}")

# ==============================================================================
# CRITICAL: Check if the lag features ARE the differences (LagT1 = diff)
# ==============================================================================
print("\n=== Checking lag feature nature (are they diffs or levels?) ===")
print(f"  Price:       mean={train['Price'].mean():.4f}, std={train['Price'].std():.4f}")
print(f"  Price_LagT1: mean={train['Price_LagT1'].mean():.4f}, std={train['Price_LagT1'].std():.4f}")
print(f"  Price_LagT2: mean={train['Price_LagT2'].mean():.4f}, std={train['Price_LagT2'].std():.4f}")
print(f"  Price_LagT3: mean={train['Price_LagT3'].mean():.4f}, std={train['Price_LagT3'].std():.4f}")

# If lag features are diffs, they'll have mean ~0 and small std
# If lag features are levels, they'll have similar mean/std to the base feature
price_diff_12 = train["Price_LagT1"] - train["Price_LagT2"]
print(f"\n  Price_LagT1 - Price_LagT2: mean={price_diff_12.mean():.6f}, std={price_diff_12.std():.6f}")

# ==============================================================================
# CRITICAL: Check if data is from order book / market microstructure
# ==============================================================================
print("\n=== Feature naming pattern analysis ===")
import pyarrow.parquet as pq
all_cols = pq.ParquetFile("train.parquet").schema.names
base_cols = [c for c in all_cols if "_Lag" not in c and c not in {"ID", "CV_GROUP", "TARGET"}]
print(f"  Base features (no lag): {len(base_cols)}")
for c in base_cols:
    print(f"    {c}")

# ==============================================================================
# CRITICAL: Try all pairs of S01/S02 features
# ==============================================================================
print("\n=== S01/S02 feature analysis (likely bid/ask) ===")
s01_cols = [c for c in base_cols if c.startswith("S01_")]
s02_cols = [c for c in base_cols if c.startswith("S02_")]
print(f"  S01 cols: {s01_cols}")
print(f"  S02 cols: {s02_cols}")

# Load S01 and S02 features
s_cols = s01_cols + s02_cols
s_data = pd.read_parquet("train.parquet", columns=s_cols + ["TARGET"])

# Try mid-price calculation
# S01 might be bid, S02 might be ask (or vice versa)
for s1 in s01_cols:
    suffix = s1.replace("S01_", "")
    s2 = f"S02_{suffix}"
    if s2 in s02_cols:
        mid = (s_data[s1] + s_data[s2]) / 2
        spread = s_data[s1] - s_data[s2]
        imbalance = s_data[s1] / (s_data[s1] + s_data[s2] + 1e-10)
        
        corr_mid = np.corrcoef(s_data["TARGET"], mid)[0, 1]
        corr_spread = np.corrcoef(s_data["TARGET"], spread)[0, 1]
        corr_imb = np.corrcoef(s_data["TARGET"], imbalance)[0, 1]
        
        r2_mid = r2_score(s_data["TARGET"], mid) if np.isfinite(mid).all() else -999
        
        print(f"  {suffix}: mid_corr={corr_mid:.6f}, spread_corr={corr_spread:.6f}, imb_corr={corr_imb:.6f}")

# ==============================================================================
# BREAKTHROUGH ATTEMPT: Try weighted sum / linear combination
# ==============================================================================
print("\n=== Trying linear regression on all base features ===")
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score

# Use all base features
all_base = pd.read_parquet("train.parquet", columns=base_cols)
X_base = all_base.values.astype(np.float32)
X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0)

# Simple ridge regression (fast)
ridge = Ridge(alpha=1.0)
groups = pd.read_parquet("train.parquet", columns=["CV_GROUP"])["CV_GROUP"].values
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)

print("  Running Ridge regression CV...")
oof_ridge = np.zeros(len(y))
for fold, (tr, va) in enumerate(gkf.split(X_base, y, groups)):
    ridge.fit(X_base[tr], y[tr])
    oof_ridge[va] = ridge.predict(X_base[va])
    fold_r2 = r2_score(y[va], oof_ridge[va])
    print(f"    Fold {fold+1}: R2={fold_r2:.6f}")

overall_r2 = r2_score(y, oof_ridge)
print(f"  Overall Ridge R2: {overall_r2:.6f}")

# Also try with all features (base + lags)
print("\n=== Trying Ridge on ALL features ===")
all_feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
# Load in chunks due to memory
X_all = np.empty((len(y), len(all_feat_cols)), dtype=np.float32)
for i in range(0, len(all_feat_cols), 50):
    batch = all_feat_cols[i:i+50]
    chunk = pd.read_parquet("train.parquet", columns=batch)
    for j, c in enumerate(batch):
        X_all[:, i+j] = chunk[c].values.astype(np.float32)
    del chunk
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

print("  Running Ridge regression CV on all features...")
oof_all = np.zeros(len(y))
for fold, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_all[tr], y[tr])
    oof_all[va] = ridge.predict(X_all[va])
    fold_r2 = r2_score(y[va], oof_all[va])
    print(f"    Fold {fold+1}: R2={fold_r2:.6f}")

overall_r2_all = r2_score(y, oof_all)
print(f"  Overall Ridge R2 (all features): {overall_r2_all:.6f}")

# Try different alpha values
for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    ridge_a = Ridge(alpha=alpha)
    oof_a = np.zeros(len(y))
    for _, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
        ridge_a.fit(X_all[tr], y[tr])
        oof_a[va] = ridge_a.predict(X_all[va])
    r2_a = r2_score(y, oof_a)
    print(f"  Ridge alpha={alpha}: R2={r2_a:.6f}")

print("\nInvestigation complete!")
