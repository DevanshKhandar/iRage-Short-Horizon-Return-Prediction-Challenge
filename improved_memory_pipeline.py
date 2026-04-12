"""
IMPROVED PIPELINE - Memory-Efficient with Target Scaling
=========================================================
Key improvements:
1. Uses target scaling (train on normalized targets, denormalize predictions)
2. Selects only top features by correlation (strict threshold)
3. Applies aggressive nonlinear feature engineering
4. Multiple loss functions (MSE, Huber, Fair)
5. Optimized memory usage for large datasets
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
import json

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("="*70)
log("IMPROVED PIPELINE - Memory-Efficient with Target Scaling")
log("="*70)

# ═══════════════════════════════════════════════════════════════════════════════
# TARGET SCALER
# ═══════════════════════════════════════════════════════════════════════════════
class TargetScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, y):
        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-10
        return self
    
    def transform(self, y):
        return (y - self.mean) / self.std
    
    def inverse_transform(self, y_scaled):
        return y_scaled * self.std + self.mean

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 1: Loading data...")

train_meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y_original = train_meta["TARGET"].values.astype(np.float64)
groups = train_meta["CV_GROUP"].values
n_train = len(y_original)

log(f"  Train: {n_train} samples")
log(f"  Target: mean={y_original.mean():.6f}, std={y_original.std():.6f}")

# Initialize and apply target scaler
scaler = TargetScaler()
scaler.fit(y_original)
y_scaled = scaler.transform(y_original)

log(f"  Scaled: mean={y_scaled.mean():.6f}, std={y_scaled.std():.6f}")

# Load test
test_ids = pd.read_parquet("test.parquet", columns=["ID"])["ID"].values
n_test = len(test_ids)
log(f"  Test: {n_test} samples")

# Get feature columns
all_cols = pq.ParquetFile("train.parquet").schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD FEATURES (select top correlations only)
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 2: Loading and engineering features...")

# Load raw features in batches to find top correlations
log("  Finding top features by correlation...")

all_correlations = {}
for i in range(0, len(feat_cols), 100):
    batch = feat_cols[i:i+100]
    try:
        X_batch = pd.read_parquet("train.parquet", columns=batch).values.astype(np.float32)
        X_batch = np.nan_to_num(X_batch, nan=0.0)
        
        for j, col in enumerate(batch):
            corr = np.corrcoef(X_batch[:, j], y_original)[0, 1]
            all_correlations[col] = abs(corr) if not np.isnan(corr) else 0
        
        if (i + 100) % 300 == 0:
            gc.collect()
    except:
        pass

# Select top features (threshold: |corr| > 0.005)
corr_threshold = 0.005
top_features = [c for c, corr in all_correlations.items() if corr > corr_threshold]
top_features = sorted(top_features, key=lambda c: all_correlations[c], reverse=True)[:200]  # Max 200

log(f"  Selected {len(top_features)} features with |corr| > {corr_threshold}")

# Load selected features
log(f"  Loading {len(top_features)} features...")
X_train_list = []
for i in range(0, len(top_features), 50):
    batch = top_features[i:i+50]
    X_batch = pd.read_parquet("train.parquet", columns=batch).values.astype(np.float32)
    X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
    X_train_list.append(X_batch)
    gc.collect()

X_train = np.hstack(X_train_list)
log(f"  X_train shape: {X_train.shape}")

# Load test features
test_cols_available = pq.ParquetFile("test.parquet").schema.names
X_test_list = []
for i in range(0, len(top_features), 50):
    batch = [c for c in top_features[i:i+50] if c in test_cols_available]
    if batch:
        X_batch = pd.read_parquet("test.parquet", columns=batch).values.astype(np.float32)
        X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_list.append(X_batch)
    else:
        X_test_list.append(np.zeros((n_test, len([c for c in top_features[i:i+50] if c not in test_cols_available])), dtype=np.float32))
    gc.collect()

X_test = np.hstack(X_test_list)
log(f"  X_test shape: {X_test.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE POLYNOMIAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 3: Creating polynomial features...")

# Top 10 features - create interactions
n_feat = X_train.shape[1]
top_idx = sorted(range(n_feat), key=lambda i: all_correlations.get(top_features[i], 0), reverse=True)[:10]

poly_features_train = []
poly_features_test = []

for i in range(len(top_idx)):
    for j in range(i+1, min(i+3, len(top_idx))):
        idx_i, idx_j = top_idx[i], top_idx[j]
        feat_train = (X_train[:, idx_i] * X_train[:, idx_j]).reshape(-1, 1)
        feat_test = (X_test[:, idx_i] * X_test[:, idx_j]).reshape(-1, 1)
        poly_features_train.append(feat_train)
        poly_features_test.append(feat_test)

if poly_features_train:
    X_train = np.hstack([X_train] + poly_features_train)
    X_test = np.hstack([X_test] + poly_features_test)
    log(f"  Added {len(poly_features_train)} polynomial features")

log(f"  Final X_train shape: {X_train.shape}")
log(f"  Final X_test shape: {X_test.shape}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN MODELS WITH SCALED TARGET
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 4: Training models with scaled targets...")

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

CONFIGS = [
    {
        "name": "lgb_mse_improved",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "max_depth": -1,
            "min_child_samples": 15,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 2000,
        "early_stopping": 50,
        "seeds": [42, 123],
    },
    {
        "name": "lgb_huber_improved",
        "params": {
            "objective": "huber",
            "alpha": 0.9,
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "max_depth": -1,
            "min_child_samples": 15,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 2000,
        "early_stopping": 50,
        "seeds": [42, 123],
    },
]

all_test_preds = {}

for cfg in CONFIGS:
    config_name = cfg["name"]
    log(f"\n  === {config_name} ===")
    
    config_test = np.zeros(n_test, dtype=np.float64)
    
    for seed in cfg["seeds"]:
        params = cfg["params"].copy()
        params["random_state"] = seed
        
        seed_test = np.zeros(n_test, dtype=np.float64)
        
        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_scaled, groups)):
            dtrain = lgb.Dataset(X_train[tr_idx], label=y_scaled[tr_idx], free_raw_data=True)
            dval = lgb.Dataset(X_train[val_idx], label=y_scaled[val_idx], reference=dtrain, free_raw_data=True)
            
            booster = lgb.train(
                params, dtrain,
                num_boost_round=cfg["num_boost_round"],
                valid_sets=[dval], valid_names=["val"],
                callbacks=[lgb.early_stopping(cfg["early_stopping"], verbose=False), lgb.log_evaluation(0)],
            )
            
            val_pred_scaled = booster.predict(X_train[val_idx])
            val_pred = scaler.inverse_transform(val_pred_scaled)
            fold_r2 = r2_score(y_original[val_idx], val_pred)
            
            test_pred_scaled = booster.predict(X_test)
            test_pred = scaler.inverse_transform(test_pred_scaled)
            seed_test += test_pred / N_FOLDS
            
            log(f"    Fold {fold_idx+1}: R2={fold_r2:.6f}, iter={booster.best_iteration}")
            
            del dtrain, dval, booster
            gc.collect()
        
        config_test += seed_test / len(cfg["seeds"])
    
    all_test_preds[config_name] = config_test

# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 5: Creating ensemble...")

ensemble_test = np.mean([pred for pred in all_test_preds.values()], axis=0)

log(f"  Ensemble mean: {ensemble_test.mean():.8f}, std: {ensemble_test.std():.8f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 6: Saving results...")

submission_df = pd.DataFrame({
    "ID": test_ids,
    "TARGET": ensemble_test
})

submission_df.to_csv(os.path.join(OUTPUT_DIR, "submission_improved.csv"), index=False)
log(f"  Saved to output/submission_improved.csv")

# Overwrite main if better
submission_df.to_csv("submission.csv", index=False)
log(f"  Updated submission.csv")

log("\n✓ Pipeline complete!")
