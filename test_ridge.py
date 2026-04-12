"""
Quick test: Can a linear model find the signal better?
The leaderboard R2 of 0.86 suggests near-perfect linear relationship.
LightGBM might be overfitting to noise with tiny trees. 
Let's try Ridge regression on all features.
"""
import gc, time, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Load data
log("Loading metadata...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = meta["TARGET"].values.astype(np.float64)
groups = meta["CV_GROUP"].values
n = len(y)
del meta; gc.collect()

test_ids = pd.read_parquet("test.parquet", columns=["ID"])["ID"].values
n_test = len(test_ids)

all_cols = pq.ParquetFile("train.parquet").schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
test_col_set = set(pq.ParquetFile("test.parquet").schema.names)

# Load ALL features into memory (445 * 661K * 4 = 1.2GB — tight but should work)
log("Loading all train features...")
X_train = np.empty((n, len(feat_cols)), dtype=np.float32)
for i in range(0, len(feat_cols), 50):
    batch = feat_cols[i:i+50]
    df = pd.read_parquet("train.parquet", columns=batch)
    for j, c in enumerate(batch):
        X_train[:, i+j] = df[c].values.astype(np.float32)
    del df
    if (i+50) % 200 == 0:
        log(f"  {min(i+50, len(feat_cols))}/{len(feat_cols)}")
        gc.collect()

X_train = np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
log(f"X_train: {X_train.shape} ({X_train.nbytes/1e9:.2f} GB)")

# Try Ridge with multiple alpha values
log("\n=== RIDGE REGRESSION CV ===")
gkf = GroupKFold(n_splits=5)

results = {}
for alpha in [0.01, 0.1, 1.0, 10, 100, 1000, 10000]:
    oof = np.zeros(n)
    fold_r2s = []
    for fi, (tr, va) in enumerate(gkf.split(X_train, y, groups)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train[tr])
        Xva = scaler.transform(X_train[va])
        ridge = Ridge(alpha=alpha)
        ridge.fit(Xtr, y[tr])
        oof[va] = ridge.predict(Xva)
        fold_r2s.append(r2_score(y[va], oof[va]))
        del Xtr, Xva; gc.collect()
    
    r2 = r2_score(y, oof)
    results[alpha] = r2
    log(f"  alpha={alpha:>8}: R2={r2:.6f}, folds={[f'{x:.6f}' for x in fold_r2s]}")
    log(f"     pred std={oof.std():.6f}, mean={oof.mean():.6f}")

best_alpha = max(results, key=results.get)
best_r2 = results[best_alpha]
log(f"\n  Best Ridge: alpha={best_alpha}, R2={best_r2:.6f}")

# Also try with clipped target
log("\n=== RIDGE WITH CLIPPED TARGET ===")
lo, hi = np.percentile(y, 1), np.percentile(y, 99)
y_clip = np.clip(y, lo, hi)

for alpha in [0.01, 0.1, 1.0, 10, 100]:
    oof = np.zeros(n)
    for fi, (tr, va) in enumerate(gkf.split(X_train, y_clip, groups)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train[tr])
        Xva = scaler.transform(X_train[va])
        ridge = Ridge(alpha=alpha)
        ridge.fit(Xtr, y_clip[tr])
        oof[va] = ridge.predict(Xva)
        del Xtr, Xva; gc.collect()
    
    r2 = r2_score(y, oof)  # Evaluate on ORIGINAL target
    log(f"  alpha={alpha:>8} (clipped train): R2={r2:.6f}, pred_std={oof.std():.6f}")

# If Ridge does well, generate submission
log("\nGenerating Ridge submission with best alpha...")
best_oof = np.zeros(n)
for fi, (tr, va) in enumerate(gkf.split(X_train, y, groups)):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train[tr])
    Xva = scaler.transform(X_train[va])
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(Xtr, y[tr])
    best_oof[va] = ridge.predict(Xva)
    del Xtr, Xva

# Free train, load test
del X_train; gc.collect()

log("Loading test features...")
X_test = np.empty((n_test, len(feat_cols)), dtype=np.float32)
for i in range(0, len(feat_cols), 50):
    batch = feat_cols[i:i+50]
    batch_avail = [c for c in batch if c in test_col_set]
    if batch_avail:
        df = pd.read_parquet("test.parquet", columns=batch_avail)
        for j, c in enumerate(batch):
            if c in test_col_set:
                X_test[:, i+j] = df[c].values.astype(np.float32)
            else:
                X_test[:, i+j] = 0.0
        del df
    gc.collect()

X_test = np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

# Train final Ridge on ALL training data
log("Training final Ridge on all data...")
scaler = StandardScaler()
X_train_all = np.empty((n, len(feat_cols)), dtype=np.float32)
for i in range(0, len(feat_cols), 50):
    batch = feat_cols[i:i+50]
    df = pd.read_parquet("train.parquet", columns=batch)
    for j, c in enumerate(batch):
        X_train_all[:, i+j] = df[c].values.astype(np.float32)
    del df; gc.collect()
X_train_all = np.nan_to_num(X_train_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_scaled = scaler.fit_transform(X_train_all)
del X_train_all; gc.collect()

ridge_final = Ridge(alpha=best_alpha)
ridge_final.fit(X_scaled, y)
del X_scaled; gc.collect()

# Predict test
X_test_scaled = scaler.transform(X_test)
test_preds = ridge_final.predict(X_test_scaled)
del X_test, X_test_scaled; gc.collect()

log(f"Test pred: mean={test_preds.mean():.6f}, std={test_preds.std():.6f}")

# Save
sub = pd.DataFrame({"ID": test_ids, "TARGET": test_preds})
sub.to_csv("submission_ridge.csv", index=False)
log(f"Saved submission_ridge.csv")
log(sub["TARGET"].describe().to_string())

# Also save the LightGBM submission as backup
log(f"\nBest Ridge R2: {best_r2:.6f}")
log(f"LightGBM R2 was: 0.000557")
log(f"If Ridge is better, use submission_ridge.csv")
log("Done!")
