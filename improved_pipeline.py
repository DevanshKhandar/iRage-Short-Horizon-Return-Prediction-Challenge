"""
Short-Horizon Return Prediction — IMPROVED Pipeline
=====================================================
Key fixes over previous pipeline:
1. USE ALL FEATURES — don't filter by linear correlation (signal is nonlinear)
2. MUCH LESS REGULARIZATION — previous model was predicting near-zero
3. HIGHER LEARNING RATE — allow the model to actually learn
4. MORE EXPRESSIVE TREES — more leaves, deeper trees
5. USE RAW TARGET — don't clip/winsorize (loses signal)
"""

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"
SEED = 42
N_FOLDS = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 1: Loading data...")

train_meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y_full = train_meta["TARGET"].values.astype(np.float32)
groups = train_meta["CV_GROUP"].values.copy()
train_ids = train_meta["ID"].values.copy()
n_train = len(y_full)
del train_meta; gc.collect()

log(f"  Train samples: {n_train}")
log(f"  Target stats: mean={y_full.mean():.6f}, std={y_full.std():.6f}")
log(f"  Target range: [{y_full.min():.4f}, {y_full.max():.4f}]")

# Get ALL feature names — DO NOT FILTER
all_schema_cols = pq.ParquetFile("train.parquet").schema.names
drop_set = {"ID", "CV_GROUP", "TARGET"}
feature_names = [c for c in all_schema_cols if c not in drop_set]
n_features = len(feature_names)
log(f"  Using ALL {n_features} features (no filtering)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD FEATURE MATRICES
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 2: Building feature matrices...")

# Load train data in chunks to manage memory
log("  Loading train features...")
CHUNK_SIZE = 50  # columns at a time
X_train = np.empty((n_train, n_features), dtype=np.float32)

for start in range(0, n_features, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, n_features)
    cols = feature_names[start:end]
    chunk = pd.read_parquet("train.parquet", columns=cols)
    for j, c in enumerate(cols):
        X_train[:, start + j] = chunk[c].values.astype(np.float32)
    del chunk
    if (start + CHUNK_SIZE) % 200 == 0 or end == n_features:
        log(f"    Loaded {end}/{n_features} features")
        gc.collect()

log(f"  X_train shape: {X_train.shape}")

# Handle NaN/inf
nan_count = np.isnan(X_train).sum()
inf_count = np.isinf(X_train).sum()
if nan_count > 0 or inf_count > 0:
    log(f"  Replacing {nan_count} NaN and {inf_count} inf values")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

# Load test data
log("  Loading test features...")
test_meta = pd.read_parquet("test.parquet", columns=["ID"])
test_ids = test_meta["ID"].values.copy()
n_test = len(test_ids)
del test_meta; gc.collect()

test_schema = pq.ParquetFile("test.parquet").schema.names
X_test = np.empty((n_test, n_features), dtype=np.float32)

for start in range(0, n_features, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, n_features)
    cols = feature_names[start:end]
    cols_available = [c for c in cols if c in test_schema]
    cols_missing = [c for c in cols if c not in test_schema]
    
    if cols_available:
        chunk = pd.read_parquet("test.parquet", columns=cols_available)
        for c in cols_available:
            idx = feature_names.index(c)
            X_test[:, idx] = chunk[c].values.astype(np.float32)
        del chunk
    
    for c in cols_missing:
        idx = feature_names.index(c)
        X_test[:, idx] = 0.0
    
    if (start + CHUNK_SIZE) % 200 == 0 or end == n_features:
        gc.collect()

X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
log(f"  X_test shape: {X_test.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN LIGHTGBM — PROPERLY TUNED
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 3: Training LightGBM with proper hyperparameters...")

# The key changes vs previous pipeline:
# - MUCH less regularization (reg_alpha 0.1 vs 5.0, reg_lambda 0.1 vs 20.0)
# - More leaves (127 vs 15-63) to capture nonlinear patterns
# - Higher learning rate (0.05 vs 0.005-0.01) 
# - Lower min_child_samples (50 vs 200-500)
# - No min_gain_to_split restriction
# - USE RAW TARGET (no winsorization)

CONFIGS = [
    {
        "name": "lgb_main",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "max_depth": -1,  # unlimited depth
            "min_child_samples": 50,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_gain_to_split": 0.0,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 5000,
        "early_stopping": 100,
        "seeds": [42, 123, 456],
        "weight": 0.5,
    },
    {
        "name": "lgb_huber",
        "params": {
            "objective": "huber",
            "alpha": 0.9,  # less aggressive than 0.5
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "max_depth": -1,
            "min_child_samples": 30,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "reg_alpha": 0.05,
            "reg_lambda": 0.05,
            "min_gain_to_split": 0.0,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 5000,
        "early_stopping": 100,
        "seeds": [42, 789, 101],
        "weight": 0.3,
    },
    {
        "name": "lgb_deep",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 511,
            "max_depth": -1,
            "min_child_samples": 20,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
            "min_gain_to_split": 0.0,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 5000,
        "early_stopping": 100,
        "seeds": [42, 202, 303],
        "weight": 0.2,
    },
]

gkf = GroupKFold(n_splits=N_FOLDS)
all_oof = {}
all_test_preds = {}
all_feature_importance = np.zeros(n_features)

for cfg in CONFIGS:
    config_name = cfg["name"]
    log(f"\n  === Config: {config_name} (weight={cfg['weight']}) ===")
    
    # USE RAW TARGET — no clipping!
    y_target = y_full
    
    config_oof = np.zeros(n_train, dtype=np.float64)
    config_test = np.zeros(n_test, dtype=np.float64)
    n_seeds = len(cfg["seeds"])
    
    for seed_idx, seed in enumerate(cfg["seeds"]):
        log(f"    Seed {seed_idx+1}/{n_seeds} (seed={seed})")
        
        params = cfg["params"].copy()
        params["random_state"] = seed
        
        seed_oof = np.zeros(n_train, dtype=np.float64)
        seed_test = np.zeros(n_test, dtype=np.float64)
        
        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_target, groups)):
            dtrain = lgb.Dataset(
                X_train[tr_idx], label=y_target[tr_idx],
                feature_name=feature_names, free_raw_data=True
            )
            dval = lgb.Dataset(
                X_train[val_idx], label=y_target[val_idx],
                feature_name=feature_names, free_raw_data=True, reference=dtrain
            )
            
            booster = lgb.train(
                params, dtrain,
                num_boost_round=cfg["num_boost_round"],
                valid_sets=[dval], valid_names=["val"],
                callbacks=[
                    lgb.early_stopping(cfg["early_stopping"], verbose=False),
                    lgb.log_evaluation(period=500),
                ],
            )
            
            val_pred = booster.predict(X_train[val_idx])
            seed_oof[val_idx] = val_pred
            seed_test += booster.predict(X_test) / N_FOLDS
            
            fold_r2 = r2_score(y_target[val_idx], val_pred)
            log(f"      Fold {fold_idx+1}: R2={fold_r2:.6f} (best_iter={booster.best_iteration})")
            
            if seed_idx == 0:
                all_feature_importance += booster.feature_importance(importance_type="gain")
            
            del dtrain, dval, booster
            gc.collect()
        
        seed_r2 = r2_score(y_full, seed_oof)
        log(f"      Seed R2: {seed_r2:.6f}")
        log(f"      Pred stats: mean={seed_oof.mean():.6f}, std={seed_oof.std():.6f}")
        
        config_oof += seed_oof / n_seeds
        config_test += seed_test / n_seeds
    
    config_r2 = r2_score(y_full, config_oof)
    log(f"    Config {config_name} OOF R2: {config_r2:.6f}")
    log(f"    Config pred range: [{config_oof.min():.6f}, {config_oof.max():.6f}]")
    
    all_oof[config_name] = config_oof
    all_test_preds[config_name] = config_test

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE BLENDING
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 4: Ensemble blending...")

config_names = [cfg["name"] for cfg in CONFIGS]

# Find optimal weights
log("  Searching for optimal weights...")
best_r2 = -999
best_weights = [cfg["weight"] for cfg in CONFIGS]

for w0 in np.arange(0.0, 1.01, 0.05):
    for w1 in np.arange(0.0, 1.01 - w0, 0.05):
        w2 = 1.0 - w0 - w1
        if w2 < 0:
            continue
        trial_oof = (all_oof[config_names[0]] * w0 + 
                     all_oof[config_names[1]] * w1 + 
                     all_oof[config_names[2]] * w2)
        trial_r2 = r2_score(y_full, trial_oof)
        if trial_r2 > best_r2:
            best_r2 = trial_r2
            best_weights = [w0, w1, w2]

log(f"  Optimal weights: {best_weights}")
log(f"  Optimal ensemble R2: {best_r2:.6f}")

# Apply optimal weights
oof_final = np.zeros(n_train, dtype=np.float64)
test_final = np.zeros(n_test, dtype=np.float64)
for i, name in enumerate(config_names):
    oof_final += all_oof[name] * best_weights[i]
    test_final += all_test_preds[name] * best_weights[i]

# Also try each config solo
for name in config_names:
    solo_r2 = r2_score(y_full, all_oof[name])
    log(f"  Solo {name}: R2={solo_r2:.6f}")

# Use best single model if it beats ensemble
for name in config_names:
    solo_r2 = r2_score(y_full, all_oof[name])
    if solo_r2 > best_r2:
        log(f"  Using solo {name} (R2={solo_r2:.6f}) over ensemble (R2={best_r2:.6f})")
        oof_final = all_oof[name]
        test_final = all_test_preds[name]
        best_r2 = solo_r2

final_r2 = r2_score(y_full, oof_final)
log(f"\n  FINAL R2: {final_r2:.6f}")
log(f"  Prediction range: [{test_final.min():.6f}, {test_final.max():.6f}]")
log(f"  Prediction std: {test_final.std():.6f}")

# Sanity check: compare with zero prediction
zero_r2 = r2_score(y_full, np.zeros(n_train))
log(f"  Baseline (predict 0) R2: {zero_r2:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. GENERATE SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 5: Generating submission...")

submission = pd.DataFrame({"ID": test_ids, "TARGET": test_final})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
submission.to_csv("submission.csv", index=False)
log(f"  Saved submission.csv ({len(submission)} rows)")
log(f"  Submission stats:")
log(f"    mean: {test_final.mean():.6f}")
log(f"    std:  {test_final.std():.6f}")
log(f"    min:  {test_final.min():.6f}")
log(f"    max:  {test_final.max():.6f}")

# Also print top feature importance
fi = all_feature_importance / (N_FOLDS * len(CONFIGS[0]["seeds"]))
fi_order = np.argsort(fi)[::-1][:20]
log("\n  Top 20 Features by Gain:")
for rank, idx in enumerate(fi_order):
    log(f"    {rank+1}. {feature_names[idx]}: {fi[idx]:.1f}")

log("\nPipeline complete!")
log(f"FINAL OOF R2: {final_r2:.6f}")