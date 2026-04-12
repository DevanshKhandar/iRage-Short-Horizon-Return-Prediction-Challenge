"""
CLUTCH FINAL - No-Filter Maximum Power LightGBM
==================================================
Previous pipelines were crippled by:
1. Dropping 85 "toxic" features (which might be the MOST predictive)
2. Only using 120 features out of 445  
3. Using weak model settings

This script uses ALL features, deeper trees, and generates multiple submissions.
"""

import gc, time, warnings, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Load metadata
# ═══════════════════════════════════════════════════════════════
log("PHASE 1: Loading metadata...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = meta["TARGET"].values.astype(np.float32)
groups = meta["CV_GROUP"].values
n_train = len(y)
del meta; gc.collect()

test_meta = pd.read_parquet("test.parquet", columns=["ID"])
n_test = len(test_meta)
log(f"  Train: {n_train}, Test: {n_test}")

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Use ALL features (no adversarial filtering!)
# ═══════════════════════════════════════════════════════════════
log("PHASE 2: Loading ALL features (no filtering)...")

all_cols = pq.ParquetFile("train.parquet").schema.names
test_cols = set(pq.ParquetFile("test.parquet").schema.names)

# Use EVERY feature that exists in both train and test
# INCLUDING SO3_T which previous pipelines wrongly excluded!
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"} and c in test_cols]
log(f"  Using {len(feat_cols)} features (previous: only 120)")

# Load train features in batches
X_train = np.empty((n_train, len(feat_cols)), dtype=np.float32)
for i in range(0, len(feat_cols), 50):
    batch = feat_cols[i:i+50]
    df = pd.read_parquet("train.parquet", columns=batch)
    for j, c in enumerate(batch):
        X_train[:, i+j] = df[c].values.astype(np.float32)
    del df; gc.collect()
    if (i+50) % 200 == 0:
        log(f"    Loaded {min(i+50, len(feat_cols))}/{len(feat_cols)} features")

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
log(f"  Train matrix: {X_train.shape}, using {X_train.nbytes / 1e9:.1f} GB")

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Train powerful LightGBM with multiple configurations
# ═══════════════════════════════════════════════════════════════
log("\nPHASE 3: Training deep LightGBM ensemble...")

# NO target clipping! The clipping was hurting performance
# Use raw target for maximum signal

N_FOLDS = 5  # Faster than 10, still robust
gkf = GroupKFold(n_splits=N_FOLDS)

CONFIGS = [
    {
        "name": "Deep_Huber",
        "params": {
            "objective": "huber", "alpha": 0.9,
            "metric": "rmse", "boosting_type": "gbdt",
            "learning_rate": 0.01, "num_leaves": 255, "max_depth": -1,
            "min_child_samples": 50,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "reg_alpha": 0.01, "reg_lambda": 0.1,
            "verbose": -1, "n_jobs": -1, "random_state": 42,
            "max_bin": 255,
        }
    },
    {
        "name": "Deep_MSE",
        "params": {
            "objective": "regression",
            "metric": "rmse", "boosting_type": "gbdt",
            "learning_rate": 0.01, "num_leaves": 255, "max_depth": -1,
            "min_child_samples": 50,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "verbose": -1, "n_jobs": -1, "random_state": 42,
            "max_bin": 255,
        }
    },
    {
        "name": "Deep_MAE",
        "params": {
            "objective": "mae",
            "metric": "mae", "boosting_type": "gbdt",
            "learning_rate": 0.01, "num_leaves": 127, "max_depth": -1,
            "min_child_samples": 80,
            "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 1,
            "reg_alpha": 0.5, "reg_lambda": 2.0,
            "verbose": -1, "n_jobs": -1, "random_state": 42,
            "max_bin": 255,
        }
    },
]

oof_preds = {}
models = {}

for cfg in CONFIGS:
    name = cfg["name"]
    log(f"  Training {name}...")
    oof_preds[name] = np.zeros(n_train, np.float32)
    models[name] = []
    
    for fi, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y, groups)):
        dt = lgb.Dataset(X_train[tr_idx], y[tr_idx], free_raw_data=True)
        dv = lgb.Dataset(X_train[va_idx], y[va_idx], free_raw_data=True, reference=dt)
        
        bst = lgb.train(
            cfg["params"], dt, 
            num_boost_round=5000,
            valid_sets=[dv], valid_names=["val"],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        oof_preds[name][va_idx] = bst.predict(X_train[va_idx])
        models[name].append(bst.model_to_string())
        
        log(f"    Fold {fi+1}/{N_FOLDS}: {bst.best_iteration} rounds")
        del dt, dv, bst; gc.collect()
    
    r2 = r2_score(y, oof_preds[name])
    log(f"  >>> {name} OOF R²: {r2:.6f} <<<")

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Optimize ensemble weights
# ═══════════════════════════════════════════════════════════════
log("\nPHASE 4: Optimizing ensemble...")

names = list(oof_preds.keys())
best_r2 = -float('inf')
best_weights = None

# Grid search weights
for w0 in np.arange(0, 1.05, 0.1):
    for w1 in np.arange(0, 1.05 - w0, 0.1):
        w2 = 1.0 - w0 - w1
        if w2 < -0.01:
            continue
        w2 = max(0, w2)
        blend = w0 * oof_preds[names[0]] + w1 * oof_preds[names[1]] + w2 * oof_preds[names[2]]
        r2 = r2_score(y, blend)
        if r2 > best_r2:
            best_r2 = r2
            best_weights = [w0, w1, w2]

log(f"  Best weights: {[f'{w:.2f}' for w in best_weights]}")
log(f"  Best blend OOF R²: {best_r2:.6f}")

# Also find optimal scale
oof_blend = sum(oof_preds[names[i]] * best_weights[i] for i in range(len(names)))
best_scale = 1.0
best_scale_r2 = best_r2

for s in np.arange(0.5, 3.0, 0.05):
    r2 = r2_score(y, oof_blend * s)
    if r2 > best_scale_r2:
        best_scale = s
        best_scale_r2 = r2

log(f"  Best scale: {best_scale:.2f}, R²: {best_scale_r2:.6f}")

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Load test features and predict
# ═══════════════════════════════════════════════════════════════
log("\nPHASE 5: Loading test features...")
del X_train; gc.collect()

X_test = np.empty((n_test, len(feat_cols)), dtype=np.float32)
for i in range(0, len(feat_cols), 50):
    batch = feat_cols[i:i+50]
    df = pd.read_parquet("test.parquet", columns=batch)
    for j, c in enumerate(batch):
        X_test[:, i+j] = df[c].values.astype(np.float32)
    del df; gc.collect()

X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
log(f"  Test matrix: {X_test.shape}")

log("  Predicting...")
test_blend = np.zeros(n_test, np.float32)

for i, name in enumerate(names):
    w = best_weights[i]
    if w == 0:
        continue
    for ms in models[name]:
        bst = lgb.Booster(model_str=ms)
        chunk = 100000
        for s in range(0, n_test, chunk):
            e = min(s + chunk, n_test)
            test_blend[s:e] += bst.predict(X_test[s:e]) * w / N_FOLDS
        del bst; gc.collect()

test_blend *= best_scale

# ═══════════════════════════════════════════════════════════════
# PHASE 6: Generate multiple submission variants
# ═══════════════════════════════════════════════════════════════
log("\nPHASE 6: Generating submissions...")

# Submission 1: Pure blend (best weights + scale)
sub1 = pd.DataFrame({"ID": test_meta["ID"].values, "TARGET": test_blend})
sub1.to_csv("submission_clutch_v1.csv", index=False)
log(f"  v1 (pure blend): mean={test_blend.mean():.8f}, std={test_blend.std():.6f}")

# Submission 2: Each model individually
for i, name in enumerate(names):
    test_single = np.zeros(n_test, np.float32)
    for ms in models[name]:
        bst = lgb.Booster(model_str=ms)
        chunk = 100000
        for s in range(0, n_test, chunk):
            e = min(s + chunk, n_test)
            test_single[s:e] += bst.predict(X_test[s:e]) / N_FOLDS
        del bst; gc.collect()
    
    r2_oof = r2_score(y, oof_preds[name])
    sub = pd.DataFrame({"ID": test_meta["ID"].values, "TARGET": test_single * best_scale})
    sub.to_csv(f"submission_clutch_{name}.csv", index=False)
    log(f"  {name}: OOF R²={r2_oof:.6f}, pred std={test_single.std():.6f}")

# Submission 3: Rank-preserve with target distribution matching
from scipy.stats import rankdata
ranks = rankdata(test_blend)
# Map ranks to quantiles of training target distribution
y_sorted = np.sort(y)
indices = ((ranks - 1) / (len(ranks) - 1) * (len(y_sorted) - 1)).astype(int)
target_matched = y_sorted[np.clip(indices, 0, len(y_sorted)-1)]
sub3 = pd.DataFrame({"ID": test_meta["ID"].values, "TARGET": target_matched})
sub3.to_csv("submission_clutch_rankmatched.csv", index=False)
log(f"  rankmatched: mean={target_matched.mean():.8f}, std={target_matched.std():.6f}")

log("\n" + "="*60)
log("ALL DONE! Generated submissions:")
log("  1. submission_clutch_v1.csv (best blend)")
log("  2. submission_clutch_Deep_Huber.csv")
log("  3. submission_clutch_Deep_MSE.csv")
log("  4. submission_clutch_Deep_MAE.csv")
log("  5. submission_clutch_rankmatched.csv")
log("="*60)

# Print top features
log("\nTop 20 features from best model:")
for name in names:
    bst = lgb.Booster(model_str=models[name][0])
    imp = bst.feature_importance(importance_type='gain')
    top_idx = np.argsort(imp)[::-1][:20]
    log(f"\n  {name}:")
    for rank, idx in enumerate(top_idx):
        log(f"    {rank+1}. {feat_cols[idx]}: {imp[idx]:.0f}")
    del bst
