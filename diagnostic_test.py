"""
Diagnostic: Test LightGBM with PROPER capacity vs over-regularized.
Quick 2-fold test to see if the model CAN learn.
"""
import gc, time, numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Load data
log("Loading data...")
train_path = "train.parquet"
meta = pd.read_parquet(train_path, columns=["TARGET", "CV_GROUP"])
y = meta["TARGET"].values.astype(np.float32)
groups = meta["CV_GROUP"].values
n = len(y)
log(f"  {n} rows, target std={y.std():.6f}")

# Load ALL features
all_cols = pq.ParquetFile(train_path).schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
log(f"  Loading {len(feat_cols)} features...")
X = pd.read_parquet(train_path, columns=feat_cols).values.astype(np.float32)
log(f"  X shape: {X.shape}, memory: {X.nbytes / 1e9:.2f} GB")

# Quick 2-fold test
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups))

# ─── Test 1: CURRENT over-regularized params ─────────────────────────────────
log("\n=== TEST 1: Over-regularized (CURRENT params) ===")
oof1 = np.zeros(n)
for fi in range(2):  # Only 2 folds for speed
    tr, va = folds[fi]
    dt = lgb.Dataset(X[tr], y[tr], feature_name=feat_cols, free_raw_data=True)
    dv = lgb.Dataset(X[va], y[va], feature_name=feat_cols, free_raw_data=True, reference=dt)
    b = lgb.train(
        {"objective": "regression", "metric": "mse", "learning_rate": 0.005,
         "num_leaves": 31, "max_depth": 5, "min_child_samples": 300,
         "feature_fraction": 0.4, "bagging_fraction": 0.6, "bagging_freq": 1,
         "reg_alpha": 2.0, "reg_lambda": 10.0, "min_gain_to_split": 0.005,
         "verbose": -1, "n_jobs": -1},
        dt, 5000, valid_sets=[dv],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof1[va] = b.predict(X[va])
    log(f"  Fold {fi+1}: best_iter={b.best_iteration}, val R2={r2_score(y[va], oof1[va]):.6f}, pred_std={oof1[va].std():.6f}")
    del dt, dv, b; gc.collect()

# ─── Test 2: PROPER capacity params ──────────────────────────────────────────
log("\n=== TEST 2: Proper capacity (LOW regularization) ===")
oof2 = np.zeros(n)
for fi in range(2):
    tr, va = folds[fi]
    dt = lgb.Dataset(X[tr], y[tr], feature_name=feat_cols, free_raw_data=True)
    dv = lgb.Dataset(X[va], y[va], feature_name=feat_cols, free_raw_data=True, reference=dt)
    b = lgb.train(
        {"objective": "regression", "metric": "mse", "learning_rate": 0.05,
         "num_leaves": 255, "max_depth": -1, "min_child_samples": 20,
         "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
         "reg_alpha": 0.01, "reg_lambda": 0.1,
         "verbose": -1, "n_jobs": -1},
        dt, 5000, valid_sets=[dv],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof2[va] = b.predict(X[va])
    log(f"  Fold {fi+1}: best_iter={b.best_iteration}, val R2={r2_score(y[va], oof2[va]):.6f}, pred_std={oof2[va].std():.6f}")
    del dt, dv, b; gc.collect()

# ─── Test 3: HIGH capacity ───────────────────────────────────────────────────
log("\n=== TEST 3: High capacity (512 leaves, no reg) ===")
oof3 = np.zeros(n)
for fi in range(2):
    tr, va = folds[fi]
    dt = lgb.Dataset(X[tr], y[tr], feature_name=feat_cols, free_raw_data=True)
    dv = lgb.Dataset(X[va], y[va], feature_name=feat_cols, free_raw_data=True, reference=dt)
    b = lgb.train(
        {"objective": "regression", "metric": "mse", "learning_rate": 0.05,
         "num_leaves": 512, "max_depth": -1, "min_child_samples": 10,
         "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 1,
         "reg_alpha": 0.0, "reg_lambda": 0.0,
         "verbose": -1, "n_jobs": -1},
        dt, 5000, valid_sets=[dv],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof3[va] = b.predict(X[va])
    log(f"  Fold {fi+1}: best_iter={b.best_iteration}, val R2={r2_score(y[va], oof3[va]):.6f}, pred_std={oof3[va].std():.6f}")
    del dt, dv, b; gc.collect()

# ─── Test 4: Huber loss with proper capacity ─────────────────────────────────
log("\n=== TEST 4: Huber loss + proper capacity ===")
oof4 = np.zeros(n)
for fi in range(2):
    tr, va = folds[fi]
    dt = lgb.Dataset(X[tr], y[tr], feature_name=feat_cols, free_raw_data=True)
    dv = lgb.Dataset(X[va], y[va], feature_name=feat_cols, free_raw_data=True, reference=dt)
    b = lgb.train(
        {"objective": "huber", "alpha": 0.9, "metric": "mse", "learning_rate": 0.05,
         "num_leaves": 255, "max_depth": -1, "min_child_samples": 20,
         "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
         "reg_alpha": 0.01, "reg_lambda": 0.1,
         "verbose": -1, "n_jobs": -1},
        dt, 5000, valid_sets=[dv],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof4[va] = b.predict(X[va])
    log(f"  Fold {fi+1}: best_iter={b.best_iteration}, val R2={r2_score(y[va], oof4[va]):.6f}, pred_std={oof4[va].std():.6f}")
    del dt, dv, b; gc.collect()

# ─── Test 5: Even higher capacity with lower LR ─────────────────────────────
log("\n=== TEST 5: 1024 leaves, lr=0.03 ===")
oof5 = np.zeros(n)
for fi in range(2):
    tr, va = folds[fi]
    dt = lgb.Dataset(X[tr], y[tr], feature_name=feat_cols, free_raw_data=True)
    dv = lgb.Dataset(X[va], y[va], feature_name=feat_cols, free_raw_data=True, reference=dt)
    b = lgb.train(
        {"objective": "regression", "metric": "mse", "learning_rate": 0.03,
         "num_leaves": 1024, "max_depth": -1, "min_child_samples": 10,
         "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
         "reg_alpha": 0.0, "reg_lambda": 0.01,
         "verbose": -1, "n_jobs": -1},
        dt, 5000, valid_sets=[dv],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    oof5[va] = b.predict(X[va])
    log(f"  Fold {fi+1}: best_iter={b.best_iteration}, val R2={r2_score(y[va], oof5[va]):.6f}, pred_std={oof5[va].std():.6f}")
    del dt, dv, b; gc.collect()

log("\n" + "="*60)
log("SUMMARY")
log("="*60)
log(f"  Test 1 (over-regularized): pred_std would show if model learned")
log(f"  Test 2 (proper capacity):  proper params")
log(f"  Test 3 (high capacity):    maximum learning")
log(f"  Test 4 (huber + capacity): robust + capacity")
log(f"  Test 5 (1024 leaves):      maximum model complexity")
log(f"\n  Target std: {y.std():.6f}")
log("DONE!")
