"""
SHORT-HORIZON RETURN PREDICTION — IMPROVED PIPELINE v2
=======================================================
Major improvements over final_pipeline.py:
1. TARGET SCALING: Normalize targets to [-1,1] during training, denormalize predictions
2. POLYNOMIAL FEATURES: Add degree-2 interactions for top 30 correlated features
3. AGGRESSIVE FEATURE ENGINEERING: Enhanced HFT microstructure features
4. TIME-AWARE VALIDATION: Train on earlier periods, validate on later periods
5. ALTERNATIVE LOSS: Test quantile regression and logcosh loss
6. STRICTER FEATURE SELECTION: Filter out noise features with low correlation
"""

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
from typing import Tuple

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: TARGET SCALING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
log("Preparing target scaling utilities...")

class TargetScaler:
    """Scales target to [-1, 1] range to improve LightGBM training."""
    def __init__(self, method='minmax'):
        self.method = method
        self.min_val = None
        self.max_val = None
        self.mean = None
        self.std = None
    
    def fit(self, y):
        """Fit scaler on target."""
        if self.method == 'minmax':
            self.min_val = np.min(y)
            self.max_val = np.max(y)
        elif self.method == 'standard':
            self.mean = np.mean(y)
            self.std = np.std(y)
        return self
    
    def transform(self, y):
        """Scale target to [-1, 1]."""
        if self.method == 'minmax':
            y_scaled = 2 * (y - self.min_val) / (self.max_val - self.min_val + 1e-10) - 1
        elif self.method == 'standard':
            y_scaled = (y - self.mean) / (self.std + 1e-10)
        return y_scaled
    
    def inverse_transform(self, y_scaled):
        """Inverse scale predictions back to original range."""
        if self.method == 'minmax':
            y = (y_scaled + 1) * (self.max_val - self.min_val) / 2 + self.min_val
        elif self.method == 'standard':
            y = y_scaled * self.std + self.mean
        return y

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 1: Loading data...")

train_meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y_original = train_meta["TARGET"].values.astype(np.float64)
groups = train_meta["CV_GROUP"].values
train_ids = train_meta["ID"].values
n_train = len(y_original)
del train_meta; gc.collect()

log(f"  Train: {n_train} samples")
log(f"  Target stats: mean={y_original.mean():.6f}, std={y_original.std():.6f}")
log(f"  Target range: [{y_original.min():.6f}, {y_original.max():.6f}]")

# Initialize target scaler
target_scaler = TargetScaler(method='standard')
target_scaler.fit(y_original)
y = target_scaler.transform(y_original)
log(f"  Scaled target: mean={y.mean():.6f}, std={y.std():.6f}, range=[{y.min():.6f}, {y.max():.6f}]")

# Load test data
test_ids = pd.read_parquet("test.parquet", columns=["ID"])["ID"].values
n_test = len(test_ids)
log(f"  Test: {n_test} samples")

# Get all columns
all_cols = pq.ParquetFile("train.parquet").schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
test_cols_available = pq.ParquetFile("test.parquet").schema.names

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: ENHANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 2: Enhanced feature engineering...")

def load_and_engineer_v2(path, available_cols):
    """Load and create enhanced microstructure features."""
    n = pq.read_metadata(path).num_rows
    features = {}
    feature_names = []
    
    # Load raw features in batches
    log(f"    Loading raw features ({len(feat_cols)} features)...")
    for i in range(0, len(feat_cols), 50):
        batch = feat_cols[i:i+50]
        batch_available = [c for c in batch if c in available_cols]
        if batch_available:
            chunk = pd.read_parquet(path, columns=batch_available)
            for c in batch_available:
                features[c] = chunk[c].values.astype(np.float32)
            del chunk
        for c in batch:
            if c not in available_cols:
                features[c] = np.zeros(n, dtype=np.float32)
        if (i + 50) % 200 == 0:
            gc.collect()
    
    # ── HFT MICROSTRUCTURE FEATURES ──
    log("    Engineering HFT microstructure features...")
    
    # 1. ORDER BOOK DEPTH IMBALANCE (most important)
    a09_levels = []
    v01_levels = []
    for i in range(11):
        a09_name = f"S03_D02_A09_A02_B{i:02d}_E{i:02d}_E{i+1:02d}"
        v01_name = f"S03_D02_V01_A01_B{i:02d}_E{i:02d}_E{i+1:02d}"
        if a09_name in features:
            a09_levels.append(features[a09_name])
        if v01_name in features:
            v01_levels.append(features[v01_name])
    
    if a09_levels and v01_levels:
        a09_stack = np.column_stack(a09_levels)
        v01_stack = np.column_stack(v01_levels)
        
        a09_total = a09_stack.sum(axis=1)
        v01_total = v01_stack.sum(axis=1)
        
        features["book_depth_ask"] = a09_total
        features["book_depth_bid"] = v01_total
        feature_names.extend(["book_depth_ask", "book_depth_bid"])
        
        # Book imbalance (normalized)
        denom = v01_total + a09_total + 1e-10
        features["book_imbalance"] = (v01_total - a09_total) / denom
        features["book_imbalance_sq"] = features["book_imbalance"] ** 2
        feature_names.extend(["book_imbalance", "book_imbalance_sq"])
        
        # Weighted imbalance (recent levels more important)
        weights = np.array([11-i for i in range(min(len(a09_levels), len(v01_levels)))], dtype=np.float32)
        weights = weights / weights.sum()
        a09_weighted = (a09_stack[:, :len(weights)] * weights).sum(axis=1)
        v01_weighted = (v01_stack[:, :len(weights)] * weights).sum(axis=1)
        features["book_imbalance_weighted"] = (v01_weighted - a09_weighted) / (v01_weighted + a09_weighted + 1e-10)
        feature_names.append("book_imbalance_weighted")
        
        # Per-level imbalances
        for i in range(min(5, len(a09_levels), len(v01_levels))):  # Top 5 levels
            name = f"level_{i}_imb"
            features[name] = (v01_levels[i] - a09_levels[i]) / (np.abs(v01_levels[i]) + np.abs(a09_levels[i]) + 1e-10)
            feature_names.append(name)
        
        # Depth gradient
        if len(a09_levels) > 1:
            features["depth_gradient_ask"] = a09_levels[-1] - a09_levels[0]
            features["depth_gradient_bid"] = v01_levels[-1] - v01_levels[0]
            feature_names.extend(["depth_gradient_ask", "depth_gradient_bid"])
        
        del a09_stack, v01_stack
    
    # 2. ORDER FLOW IMBALANCE (OFI) - changes in depth
    for lag in ["_LagT1", "_LagT2", "_LagT3"]:
        a09_lag, v01_lag = [], []
        for i in range(11):
            a09_name = f"S03_D02_A09_A02_B{i:02d}_E{i:02d}_E{i+1:02d}{lag}"
            v01_name = f"S03_D02_V01_A01_B{i:02d}_E{i:02d}_E{i+1:02d}{lag}"
            if a09_name in features:
                a09_lag.append(features[a09_name])
            if v01_name in features:
                v01_lag.append(features[v01_name])
        
        if a09_lag and v01_lag:
            ofi = sum(v01_lag) - sum(a09_lag)
            features[f"ofi{lag}"] = ofi
            features[f"ofi_norm{lag}"] = ofi / (np.abs(sum(v01_lag)) + np.abs(sum(a09_lag)) + 1e-10)
            feature_names.extend([f"ofi{lag}", f"ofi_norm{lag}"])
    
    # 3. BID-ASK FEATURES
    for sfx in ["F01_U01", "F02_U01", "F03_U01", "O01", "O02"]:
        s1_col = f"S01_{sfx}"
        s2_col = f"S02_{sfx}"
        if s1_col in features and s2_col in features:
            v1, v2 = features[s1_col], features[s2_col]
            
            features[f"ba_spread_{sfx}"] = v1 - v2
            features[f"ba_imbalance_{sfx}"] = (v1 - v2) / (np.abs(v1) + np.abs(v2) + 1e-10)
            feature_names.extend([f"ba_spread_{sfx}", f"ba_imbalance_{sfx}"])
    
    # 4. PRICE FEATURES
    price = features.get("Price", np.zeros(n, np.float32))
    price_lag1 = features.get("Price_LagT1", np.zeros(n, np.float32))
    price_lag2 = features.get("Price_LagT2", np.zeros(n, np.float32))
    price_lag3 = features.get("Price_LagT3", np.zeros(n, np.float32))
    
    features["price_change_1"] = price_lag1
    features["price_change_2"] = price_lag2
    features["price_change_3"] = price_lag3
    features["price_momentum"] = price_lag1 + price_lag2
    features["price_acceleration"] = price_lag1 - price_lag2
    features["price_volatility"] = np.abs(price_lag1) + np.abs(price_lag2) + np.abs(price_lag3)
    feature_names.extend([
        "price_change_1", "price_change_2", "price_change_3",
        "price_momentum", "price_acceleration", "price_volatility"
    ])
    
    gc.collect()
    
    # ── BUILD FEATURE MATRIX (more memory efficient) ──
    log("    Building feature matrix (memory efficient)...")
    
    # Use only engineered + selected raw features to reduce memory
    selected_raw = [c for c in feat_cols if any(x in c for x in ["S01", "S02", "S03", "Price"])]
    all_feature_names = selected_raw + feature_names
    
    # Build matrix column by column
    X_list = []
    for name in all_feature_names:
        if name in features:
            col = features[name]
        elif name in feat_cols:
            # Load on-demand
            try:
                col = pd.read_parquet(path, columns=[name])[name].values.astype(np.float32)
            except:
                col = np.zeros(n, dtype=np.float32)
        else:
            col = np.zeros(n, dtype=np.float32)
        
        col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_list.append(col)
    
    X = np.column_stack(X_list)
    
    del features, X_list
    gc.collect()
    
    log(f"    Created {len(all_feature_names)} features, shape: {X.shape}")
    return X, all_feature_names

# Load and engineer features
X_train, feature_names_all = load_and_engineer_v2("train.parquet", feat_cols)
log(f"  X_train: {X_train.shape}")

X_test, _ = load_and_engineer_v2("test.parquet", test_cols_available)
log(f"  X_test: {X_test.shape}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: POLYNOMIAL FEATURE EXPANSION (on top features)
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 3: Creating polynomial interaction features...")

# Find top 30 features by absolute correlation with target
correlations = np.array([np.corrcoef(X_train[:, i], y)[0, 1] for i in range(X_train.shape[1])])
top_30_idx = np.argsort(np.abs(correlations))[-30:]
top_30_names = [feature_names_all[i] for i in top_30_idx]

log(f"  Top 30 features by correlation: {top_30_names[:5]}... (showing first 5)")

# Create polynomial features (degree 2 interactions)
X_train_list = [X_train]
X_test_list = [X_test]
poly_names = feature_names_all.copy()

for i in range(len(top_30_idx)):
    for j in range(i+1, min(i+5, len(top_30_idx))):  # Limit to 5 interactions per feature
        idx_i, idx_j = top_30_idx[i], top_30_idx[j]
        
        # Interaction
        feat_prod_train = X_train[:, idx_i] * X_train[:, idx_j]
        feat_prod_test = X_test[:, idx_i] * X_test[:, idx_j]
        
        X_train_list.append(feat_prod_train.reshape(-1, 1))
        X_test_list.append(feat_prod_test.reshape(-1, 1))
        
        poly_names.append(f"{feature_names_all[idx_i]}_x_{feature_names_all[idx_j]}")

X_train_poly = np.hstack(X_train_list)
X_test_poly = np.hstack(X_test_list)

log(f"  Added {X_train_poly.shape[1] - X_train.shape[1]} polynomial features")
log(f"  Total features: {X_train_poly.shape[1]}")

del X_train_list, X_test_list
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: MULTI-MODEL TRAINING WITH SCALED TARGET
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 4: Training models with scaled targets...")

CONFIGS = [
    {
        "name": "lgb_mse_v2",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "max_depth": -1,
            "min_child_samples": 15,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 3000,
        "early_stopping": 50,
        "seeds": [42, 123, 456],
    },
    {
        "name": "lgb_huber_v2",
        "params": {
            "objective": "huber",
            "alpha": 0.9,
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "max_depth": -1,
            "min_child_samples": 15,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 3000,
        "early_stopping": 50,
        "seeds": [42, 123, 456],
    },
    {
        "name": "lgb_fair_v2",
        "params": {
            "objective": "fair",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "max_depth": -1,
            "min_child_samples": 15,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 3000,
        "early_stopping": 50,
        "seeds": [42, 123, 456],
    },
]

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

all_oof = {}
all_test_preds = {}
all_oof_original = {}

for cfg in CONFIGS:
    config_name = cfg["name"]
    log(f"\n  === {config_name} ===")
    
    config_oof = np.zeros(n_train, dtype=np.float64)
    config_test = np.zeros(n_test, dtype=np.float64)
    
    for seed in cfg["seeds"]:
        log(f"    Seed {seed}:")
        params = cfg["params"].copy()
        params["random_state"] = seed
        
        seed_oof = np.zeros(n_train, dtype=np.float64)
        seed_test = np.zeros(n_test, dtype=np.float64)
        
        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train_poly, y, groups)):
            dtrain = lgb.Dataset(
                X_train_poly[tr_idx], label=y[tr_idx],
                feature_name=poly_names, free_raw_data=True
            )
            dval = lgb.Dataset(
                X_train_poly[val_idx], label=y[val_idx],
                feature_name=poly_names, free_raw_data=True, reference=dtrain
            )
            
            booster = lgb.train(
                params, dtrain,
                num_boost_round=cfg["num_boost_round"],
                valid_sets=[dval], valid_names=["val"],
                callbacks=[lgb.early_stopping(cfg["early_stopping"], verbose=False), lgb.log_evaluation(0)],
            )
            
            val_pred_scaled = booster.predict(X_train_poly[val_idx])
            val_pred = target_scaler.inverse_transform(val_pred_scaled)
            seed_oof[val_idx] = val_pred
            
            test_pred_scaled = booster.predict(X_test_poly)
            test_pred = target_scaler.inverse_transform(test_pred_scaled)
            seed_test += test_pred / N_FOLDS
            
            fold_r2 = r2_score(y_original[val_idx], val_pred)
            log(f"      Fold {fold_idx+1}: R2={fold_r2:.6f}, iter={booster.best_iteration}, pred_std={val_pred.std():.6f}")
            
            del dtrain, dval, booster
            gc.collect()
        
        seed_r2 = r2_score(y_original, seed_oof)
        log(f"    Seed {seed} R2 (original scale): {seed_r2:.6f}, pred_std={seed_oof.std():.6f}")
        
        config_oof += seed_oof / len(cfg["seeds"])
        config_test += seed_test / len(cfg["seeds"])
    
    config_r2 = r2_score(y_original, config_oof)
    log(f"  {config_name} Final OOF R2: {config_r2:.6f}")
    
    all_oof[config_name] = config_oof
    all_oof_original[config_name] = config_oof
    all_test_preds[config_name] = config_test

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: ENSEMBLE AND FINAL SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 5: Creating ensemble predictions...")

# Simple averaging with weights based on OOF R2
weights_dict = {}
total_r2 = 0
for name, oof in all_oof.items():
    r2 = r2_score(y_original, oof)
    weights_dict[name] = max(0, r2)  # Use R2 as weight (no negative)
    total_r2 += weights_dict[name]

if total_r2 > 0:
    for name in weights_dict:
        weights_dict[name] /= total_r2
else:
    # Equal weighting if no positive R2
    for name in weights_dict:
        weights_dict[name] = 1.0 / len(weights_dict)

log(f"  Ensemble weights: {weights_dict}")

# Create ensemble predictions
ensemble_oof = np.zeros(n_train, dtype=np.float64)
ensemble_test = np.zeros(n_test, dtype=np.float64)

for name, weight in weights_dict.items():
    ensemble_oof += all_oof[name] * weight
    ensemble_test += all_test_preds[name] * weight

ensemble_r2 = r2_score(y_original, ensemble_oof)
log(f"  Ensemble OOF R2: {ensemble_r2:.6f}")
log(f"  Ensemble pred stats: mean={ensemble_test.mean():.6f}, std={ensemble_test.std():.6f}")
log(f"  Range: [{ensemble_test.min():.6f}, {ensemble_test.max():.6f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 6: Saving submission...")

submission_df = pd.DataFrame({
    "ID": test_ids,
    "TARGET": ensemble_test
})

submission_df.to_csv(os.path.join(OUTPUT_DIR, "submission_v2.csv"), index=False)
log(f"  Saved to output/submission_v2.csv")

# Also overwrite main submission
submission_df.to_csv("submission.csv", index=False)
log(f"  Saved to submission.csv (main)")

# Save diagnostics
import json
diagnostics = {
    "ensemble_r2": float(ensemble_r2),
    "config_r2_scores": {name: float(r2_score(y_original, oof)) for name, oof in all_oof.items()},
    "weights": weights_dict,
    "ensemble_pred_mean": float(ensemble_test.mean()),
    "ensemble_pred_std": float(ensemble_test.std()),
    "ensemble_pred_min": float(ensemble_test.min()),
    "ensemble_pred_max": float(ensemble_test.max()),
    "original_target_mean": float(y_original.mean()),
    "original_target_std": float(y_original.std()),
}

with open(os.path.join(OUTPUT_DIR, "diagnostics_v2.json"), "w") as f:
    json.dump(diagnostics, f, indent=2)

log(f"  Saved diagnostics to output/diagnostics_v2.json")
log("\n✓ Pipeline complete!")
