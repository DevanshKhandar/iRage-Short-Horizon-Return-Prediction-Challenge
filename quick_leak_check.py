import pandas as pd, numpy as np
from sklearn.metrics import r2_score

print("Loading SO3_T from both...")
train = pd.read_parquet("train.parquet", columns=["SO3_T", "TARGET", "CV_GROUP", "ID", "Price"])
test = pd.read_parquet("test.parquet", columns=["SO3_T", "ID", "Price"])

# Check exact SO3_T overlaps
train_so3t = set(np.round(train["SO3_T"].values, 10))
test_so3t = set(np.round(test["SO3_T"].values, 10))
overlap = train_so3t & test_so3t
print(f"Exact SO3_T overlap: {len(overlap)} values")
print(f"Train unique SO3_T: {len(train_so3t)}")
print(f"Test unique SO3_T: {len(test_so3t)}")

if len(overlap) > 0:
    print(f"OVERLAP EXISTS! Percentage of test SO3_T values in train: {len(overlap)/len(test_so3t)*100:.2f}%")
    
    # Match test rows to train rows via SO3_T
    train_lookup = train.groupby(np.round(train["SO3_T"], 10))["TARGET"].mean().to_dict()
    test["SO3_T_round"] = np.round(test["SO3_T"], 10)
    test["matched_target"] = test["SO3_T_round"].map(train_lookup)
    matched = test["matched_target"].notna()
    print(f"Test rows matched: {matched.sum()}/{len(test)} ({matched.mean()*100:.2f}%)")
    
    if matched.sum() > 0:
        mt = test.loc[matched, "matched_target"]
        print(f"Matched targets: mean={mt.mean():.8f}, std={mt.std():.6f}")

# Check within-group structure
print("\n=== Per-group target analysis ===")
print("Rows per CV_GROUP (top 5):")
for g in range(min(5, train["CV_GROUP"].nunique())):
    mask = train["CV_GROUP"] == g
    print(f"  Group {g}: n={mask.sum()}, target mean={train.loc[mask, 'TARGET'].mean():.8f}, std={train.loc[mask, 'TARGET'].std():.6f}")

# Check if SO3_T encodes group info
print("\n=== SO3_T distribution per CV_GROUP ===")
for g in range(5):
    mask = train["CV_GROUP"] == g
    so3 = train.loc[mask, "SO3_T"]
    print(f"  Group {g}: SO3_T range=[{so3.min():.8f}, {so3.max():.8f}], mean={so3.mean():.8f}")

# CRITICAL: Check per-group per-SO3_T consistency
print("\n=== Same SO3_T across groups check ===")
train["SO3_T_bin"] = np.round(train["SO3_T"], 6)
multi_so3t = train.groupby("SO3_T_bin").agg(
    n_groups=("CV_GROUP", "nunique"),
    count=("TARGET", "count"),
    target_std=("TARGET", "std")
).reset_index()

multi = multi_so3t[multi_so3t["n_groups"] > 1]
print(f"SO3_T bins spanning >1 CV_GROUP: {len(multi)}")
if len(multi) > 0:
    print(f"Avg groups per multi-bin: {multi['n_groups'].mean():.1f}")
    print(f"Max groups in a bin: {multi['n_groups'].max()}")
    print(f"Avg target std within multi-bins: {multi['target_std'].mean():.6f}")

# Check if CV_GROUP can be predicted from features for test set
# If we can infer CV_GROUP for test, we might be able to use group-specific models
print("\n=== Can we infer CV_GROUP from Price? ===")
price_by_group = train.groupby("CV_GROUP")["Price"].agg(["mean", "std", "min", "max"])
print(price_by_group.head(10))
# Are groups separable by Price range?
print(f"\nPrice range overlaps between groups (first 10):")
for g in range(min(10, len(price_by_group))):
    row = price_by_group.iloc[g]
    print(f"  Group {g}: [{row['min']:.0f}, {row['max']:.0f}]")

# Check Price distinctness
print("\n=== Price overlap between train and test ===")
train_prices = set(np.round(train["Price"].values, 4))
test_prices = set(np.round(test["Price"].values, 4))
price_overlap = train_prices & test_prices
print(f"Price overlap: {len(price_overlap)} unique train prices also in test")
print(f"Pct of test prices in train: {len(price_overlap)/len(test_prices)*100:.2f}%")
