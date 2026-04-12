"""
Short-Horizon Return Prediction — FINAL PIPELINE
==================================================
Key insights from data investigation:
1. This is HFT order book data from iRage Capital
2. LagT1/T2/T3 features are DIFFERENCES (changes), not levels
3. S01/S02 = bid/ask sides; S03_D02 = order book depth at 11 levels
4. Signal is in MICROSTRUCTURE features: OFI, book imbalance, microprice
5. Individual correlations are weak (<0.03) but the signal is in interactions
6. Need aggressive feature engineering + proper model settings
"""

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 1: Loading data...")

# Load metadata
train_meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = train_meta["TARGET"].values.astype(np.float64)
groups = train_meta["CV_GROUP"].values
train_ids = train_meta["ID"].values
n_train = len(y)
del train_meta; gc.collect()

log(f"  Train: {n_train} samples, target std={y.std():.6f}")

# Get all column names
all_cols = pq.ParquetFile("train.parquet").schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
test_cols_available = pq.ParquetFile("test.parquet").schema.names

# Load test IDs
test_ids = pd.read_parquet("test.parquet", columns=["ID"])["ID"].values
n_test = len(test_ids)
log(f"  Test: {n_test} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MICROSTRUCTURE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 2: Engineering microstructure features...")

def load_and_engineer(path, available_cols):
    """Load data and create HFT microstructure features."""
    n = pq.read_metadata(path).num_rows
    features = {}
    feature_names = []
    
    # ── Load all raw features ──
    log(f"    Loading raw features from {path}...")
    for i in range(0, len(feat_cols), 50):
        batch = feat_cols[i:i+50]
        batch_available = [c for c in batch if c in available_cols]
        if batch_available:
            chunk = pd.read_parquet(path, columns=batch_available)
            for c in batch_available:
                features[c] = chunk[c].values.astype(np.float32)
            del chunk
        # Fill missing columns with zeros
        for c in batch:
            if c not in available_cols:
                features[c] = np.zeros(n, dtype=np.float32)
        if (i + 50) % 200 == 0:
            gc.collect()
    
    log(f"    Loaded {len(features)} raw features")
    
    # ── FEATURE GROUP 1: Bid-Ask features (S01 vs S02) ──
    log("    Engineering bid-ask features...")
    
    # S01 = bid side, S02 = ask side
    s1_suffixes = ["F01_U01", "F02_U01", "F03_U01", "O01", "O02", "O01_A01", "O02_A01"]
    
    for sfx in s1_suffixes:
        s1 = f"S01_{sfx}"
        s2 = f"S02_{sfx}"
        if s1 in features and s2 in features:
            v1, v2 = features[s1], features[s2]
            
            # Spread
            name = f"spread_{sfx}"
            features[name] = v1 - v2
            feature_names.append(name)
            
            # Mid
            name = f"mid_{sfx}"
            features[name] = (v1 + v2) / 2
            feature_names.append(name)
            
            # Imbalance (normalized)
            name = f"imbalance_{sfx}"
            denom = np.abs(v1) + np.abs(v2) + 1e-10
            features[name] = (v1 - v2) / denom
            feature_names.append(name)
            
            # Log ratio
            name = f"logratio_{sfx}"
            features[name] = np.log((np.abs(v1) + 1e-10) / (np.abs(v2) + 1e-10))
            feature_names.append(name)
    
    # Cross-side interactions for lag features too
    for lag in ["_LagT1", "_LagT2", "_LagT3"]:
        for sfx in s1_suffixes:
            s1 = f"S01_{sfx}{lag}"
            s2 = f"S02_{sfx}{lag}"
            if s1 in features and s2 in features:
                v1, v2 = features[s1], features[s2]
                name = f"spread_{sfx}{lag}"
                features[name] = v1 - v2
                feature_names.append(name)
                name = f"imbalance_{sfx}{lag}"
                denom = np.abs(v1) + np.abs(v2) + 1e-10
                features[name] = (v1 - v2) / denom
                feature_names.append(name)
    
    # ── FEATURE GROUP 2: Order Book Depth Imbalance ──
    log("    Engineering order book depth features...")
    
    # S03_D02_A09_A02_B{i}_E{i}_E{i+1} — one side of the book
    # S03_D02_V01_A01_B{i}_E{i}_E{i+1} — other side of the book
    
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
        
        # Total depth each side
        a09_total = a09_stack.sum(axis=1)
        v01_total = v01_stack.sum(axis=1)
        features["book_total_a09"] = a09_total
        features["book_total_v01"] = v01_total
        feature_names.extend(["book_total_a09", "book_total_v01"])
        
        # Book imbalance (most important HFT feature!)
        features["book_imbalance"] = (v01_total - a09_total) / (v01_total + a09_total + 1e-10)
        feature_names.append("book_imbalance")
        
        # Weighted book imbalance (weight by level proximity)
        weights = np.array([11-i for i in range(min(len(a09_levels), len(v01_levels)))], dtype=np.float32)
        weights = weights / weights.sum()
        a09_weighted = (a09_stack[:, :len(weights)] * weights).sum(axis=1)
        v01_weighted = (v01_stack[:, :len(weights)] * weights).sum(axis=1)
        features["book_imbalance_weighted"] = (v01_weighted - a09_weighted) / (v01_weighted + a09_weighted + 1e-10)
        feature_names.append("book_imbalance_weighted")
        
        # Level-by-level imbalance
        for i in range(min(len(a09_levels), len(v01_levels))):
            name = f"level_{i}_imbalance"
            features[name] = (v01_levels[i] - a09_levels[i]) / (np.abs(v01_levels[i]) + np.abs(a09_levels[i]) + 1e-10)
            feature_names.append(name)
        
        # Depth gradient (how depth changes across levels)
        if len(a09_levels) > 1:
            features["a09_depth_gradient"] = a09_levels[-1] - a09_levels[0]
            features["v01_depth_gradient"] = v01_levels[-1] - v01_levels[0]
            feature_names.extend(["a09_depth_gradient", "v01_depth_gradient"])
        
        del a09_stack, v01_stack
    
    # Same for lag features (Order Flow Imbalance = change in depth)
    for lag in ["_LagT1", "_LagT2", "_LagT3"]:
        a09_lag = []
        v01_lag = []
        for i in range(11):
            a09_name = f"S03_D02_A09_A02_B{i:02d}_E{i:02d}_E{i+1:02d}{lag}"
            v01_name = f"S03_D02_V01_A01_B{i:02d}_E{i:02d}_E{i+1:02d}{lag}"
            if a09_name in features:
                a09_lag.append(features[a09_name])
            if v01_name in features:
                v01_lag.append(features[v01_name])
        
        if a09_lag and v01_lag:
            a09_sum = sum(a09_lag)
            v01_sum = sum(v01_lag)
            
            # OFI = change in bid depth - change in ask depth
            ofi_name = f"ofi{lag}"
            features[ofi_name] = v01_sum - a09_sum
            feature_names.append(ofi_name)
            
            # Normalized OFI
            ofi_norm_name = f"ofi_norm{lag}"
            features[ofi_norm_name] = (v01_sum - a09_sum) / (np.abs(v01_sum) + np.abs(a09_sum) + 1e-10)
            feature_names.append(ofi_norm_name)
    
    # ── FEATURE GROUP 3: Price-based features ──
    log("    Engineering price features...")
    
    price = features.get("Price", np.zeros(n, np.float32))
    price_lag1 = features.get("Price_LagT1", np.zeros(n, np.float32))
    price_lag2 = features.get("Price_LagT2", np.zeros(n, np.float32))
    price_lag3 = features.get("Price_LagT3", np.zeros(n, np.float32))
    
    # Return features (lag features are diffs, so lag1/price ≈ return)
    features["return_lag1"] = price_lag1 / (price + 1e-10)
    features["return_lag2"] = price_lag2 / (price + 1e-10)
    features["return_lag3"] = price_lag3 / (price + 1e-10)
    feature_names.extend(["return_lag1", "return_lag2", "return_lag3"])
    
    # Momentum (sum of returns)
    features["momentum_12"] = price_lag1 + price_lag2
    features["momentum_123"] = price_lag1 + price_lag2 + price_lag3
    feature_names.extend(["momentum_12", "momentum_123"])
    
    # Reversal (acceleration)
    features["acceleration"] = price_lag1 - price_lag2
    features["jerk"] = (price_lag1 - price_lag2) - (price_lag2 - price_lag3)
    feature_names.extend(["acceleration", "jerk"])
    
    # Volatility proxy
    features["vol_proxy"] = np.abs(price_lag1) + np.abs(price_lag2) + np.abs(price_lag3)
    feature_names.append("vol_proxy")
    
    # ── FEATURE GROUP 4: Volume/Tick features ──
    log("    Engineering volume/tick features...")
    
    # S03_V02-V08 and T01-T06 features — compute cross-ratios
    for v_prefix in ["V02", "V03", "V04", "V05"]:
        for t_suffix in ["T01", "T02", "T03", "T04", "T05", "T06"]:
            col = f"S03_{v_prefix}_{t_suffix}"
            if col in features:
                # Current change vs lag1 change  
                lag1 = f"{col}_LagT1"
                if lag1 in features:
                    name = f"{v_prefix}_{t_suffix}_momentum"
                    features[name] = features[col] - features[lag1]
                    feature_names.append(name)
    
    # V06/V07/V08 features
    for col_name in ["S03_V06_V01", "S03_V07_V06", "S03_V08_V06"]:
        if col_name in features:
            lag1 = f"{col_name}_LagT1"
            if lag1 in features:
                name = f"{col_name.replace('S03_', '')}_change"
                features[name] = features[col_name] - features[lag1]
                feature_names.append(name)
    
    # ── FEATURE GROUP 5: Cross-feature interactions ──
    log("    Engineering cross-feature interactions...")
    
    # Price * book imbalance
    if "book_imbalance" in features:
        features["price_x_imbalance"] = price * features["book_imbalance"]
        feature_names.append("price_x_imbalance")
    
    # Book imbalance * momentum
    if "book_imbalance" in features and "momentum_12" in features:
        features["imbalance_x_momentum"] = features["book_imbalance"] * features["momentum_12"]
        feature_names.append("imbalance_x_momentum")
    
    # OFI * price
    if "ofi_LagT1" in features:
        features["ofi1_x_price"] = features["ofi_LagT1"] * price
        feature_names.append("ofi1_x_price")
    
    # S01/S02 spread * book imbalance
    if "spread_O02" in features and "book_imbalance" in features:
        features["spread_x_imbalance"] = features["spread_O02"] * features["book_imbalance"]
        feature_names.append("spread_x_imbalance")
    
    # SO3_T (timestamp) features
    so3t = features.get("SO3_T", np.zeros(n, np.float32))
    features["so3t_sq"] = so3t ** 2
    features["so3t_inv"] = 1.0 / (so3t + 1e-10)
    feature_names.extend(["so3t_sq", "so3t_inv"])
    
    # S04/S05 features interactions
    for c in ["S04_V19_V12", "S04_V19_A06", "S05_V19"]:
        if c in features:
            lag1 = f"{c}_LagT1"
            if lag1 in features:
                name = f"{c.replace('S0', 'sec')}_change"
                features[name] = features[c] - features[lag1]
                feature_names.append(name)
    
    # ── FEATURE GROUP 6: Aggregate statistics ──
    log("    Engineering aggregate features...")
    
    # Mean/std of all current-level S03 features
    s03_base = [c for c in feat_cols if c.startswith("S03_") and "_Lag" not in c]
    if s03_base:
        s03_vals = np.column_stack([features.get(c, np.zeros(n, np.float32)) for c in s03_base])
        features["s03_mean"] = np.nanmean(s03_vals, axis=1).astype(np.float32)
        features["s03_std"] = np.nanstd(s03_vals, axis=1).astype(np.float32)
        features["s03_skew"] = np.zeros(n, np.float32)  # placeholder
        feature_names.extend(["s03_mean", "s03_std", "s03_skew"])
        del s03_vals
    
    gc.collect()
    
    # ── Combine all features into matrix ──
    # Use raw features + engineered features
    all_feature_names = list(feat_cols) + feature_names
    X = np.empty((n, len(all_feature_names)), dtype=np.float32)
    
    for i, name in enumerate(all_feature_names):
        if name in features:
            X[:, i] = features[name]
        else:
            X[:, i] = 0.0
    
    # Clean up
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    del features
    gc.collect()
    
    log(f"    Total features: {len(all_feature_names)}")
    return X, all_feature_names

X_train, feature_names_all = load_and_engineer("train.parquet", feat_cols)
log(f"  X_train: {X_train.shape}")

X_test, _ = load_and_engineer("test.parquet", test_cols_available)
log(f"  X_test: {X_test.shape}")

gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 3: Training models...")

# Configuration: try multiple approaches
CONFIGS = [
    {
        "name": "lgb_aggressive",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "max_depth": -1,
            "min_child_samples": 20,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "min_gain_to_split": 0.0,
            "verbose": -1,
            "n_jobs": -1,
            "max_bin": 255,
        },
        "num_boost_round": 3000,
        "early_stopping": 50,
        "seeds": [42],
    },
    {
        "name": "lgb_wide",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 512,
            "max_depth": -1,
            "min_child_samples": 10,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "min_gain_to_split": 0.0,
            "verbose": -1,
            "n_jobs": -1,
            "max_bin": 511,
        },
        "num_boost_round": 3000,
        "early_stopping": 50,
        "seeds": [42],
    },
    {
        "name": "lgb_dart",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "dart",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "max_depth": -1,
            "min_child_samples": 30,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 0.1,
            "drop_rate": 0.1,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 500,  # DART is slow
        "early_stopping": 50,
        "seeds": [42],
    },
    {
        "name": "lgb_huber_agg",
        "params": {
            "objective": "huber",
            "alpha": 0.9,
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "max_depth": -1,
            "min_child_samples": 20,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 3000,
        "early_stopping": 50,
        "seeds": [42],
    },
]

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

all_oof = {}
all_test_preds = {}

for cfg in CONFIGS:
    config_name = cfg["name"]
    log(f"\n  === Config: {config_name} ===")
    
    config_oof = np.zeros(n_train, dtype=np.float64)
    config_test = np.zeros(n_test, dtype=np.float64)
    
    for seed in cfg["seeds"]:
        params = cfg["params"].copy()
        params["random_state"] = seed
        
        seed_oof = np.zeros(n_train, dtype=np.float64)
        seed_test = np.zeros(n_test, dtype=np.float64)
        
        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y, groups)):
            dtrain = lgb.Dataset(
                X_train[tr_idx], label=y[tr_idx],
                feature_name=feature_names_all, free_raw_data=True
            )
            dval = lgb.Dataset(
                X_train[val_idx], label=y[val_idx],
                feature_name=feature_names_all, free_raw_data=True, reference=dtrain
            )
            
            callbacks = [
                lgb.early_stopping(cfg["early_stopping"], verbose=False),
                lgb.log_evaluation(period=0),
            ]
            
            booster = lgb.train(
                params, dtrain,
                num_boost_round=cfg["num_boost_round"],
                valid_sets=[dval], valid_names=["val"],
                callbacks=callbacks,
            )
            
            val_pred = booster.predict(X_train[val_idx])
            seed_oof[val_idx] = val_pred
            seed_test += booster.predict(X_test) / N_FOLDS
            
            fold_r2 = r2_score(y[val_idx], val_pred)
            log(f"    Fold {fold_idx+1}: R2={fold_r2:.6f}, best_iter={booster.best_iteration}, pred_std={val_pred.std():.6f}")
            
            del dtrain, dval, booster
            gc.collect()
        
        seed_r2 = r2_score(y, seed_oof)
        log(f"    Seed {seed} R2: {seed_r2:.6f}")
        
        config_oof += seed_oof / len(cfg["seeds"])
        config_test += seed_test / len(cfg["seeds"])
    
    config_r2 = r2_score(y, config_oof)
    log(f"  Config {config_name} OOF R2: {config_r2:.6f}")
    log(f"  Pred range: [{config_oof.min():.6f}, {config_oof.max():.6f}], std={config_oof.std():.6f}")
    
    all_oof[config_name] = config_oof
    all_test_preds[config_name] = config_test

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ALSO TRY: Regular KFold (ignore groups)
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 4: Training with regular KFold (no group constraint)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
params_kf = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

kf_oof = np.zeros(n_train, dtype=np.float64)
kf_test = np.zeros(n_test, dtype=np.float64)

for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    dtrain = lgb.Dataset(X_train[tr_idx], label=y[tr_idx], feature_name=feature_names_all, free_raw_data=True)
    dval = lgb.Dataset(X_train[val_idx], label=y[val_idx], feature_name=feature_names_all, free_raw_data=True, reference=dtrain)
    
    booster = lgb.train(
        params_kf, dtrain, num_boost_round=3000,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    
    kf_oof[val_idx] = booster.predict(X_train[val_idx])
    kf_test += booster.predict(X_test) / 5
    
    fold_r2 = r2_score(y[val_idx], kf_oof[val_idx])
    log(f"  KFold {fold_idx+1}: R2={fold_r2:.6f}, best_iter={booster.best_iteration}")
    
    del dtrain, dval, booster
    gc.collect()

kf_r2 = r2_score(y, kf_oof)
log(f"  KFold OOF R2: {kf_r2:.6f}")
all_oof["lgb_kfold"] = kf_oof
all_test_preds["lgb_kfold"] = kf_test

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ALSO TRY: XGBoost if available
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import xgboost as xgb
    log("\nStep 5: Training XGBoost...")
    
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0,
    }
    
    xgb_oof = np.zeros(n_train, dtype=np.float64)
    xgb_test = np.zeros(n_test, dtype=np.float64)
    
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y, groups)):
        dtrain = xgb.DMatrix(X_train[tr_idx], label=y[tr_idx], feature_names=feature_names_all)
        dval = xgb.DMatrix(X_train[val_idx], label=y[val_idx], feature_names=feature_names_all)
        dtest = xgb.DMatrix(X_test, feature_names=feature_names_all)
        
        booster = xgb.train(
            xgb_params, dtrain, num_boost_round=3000,
            evals=[(dval, "val")],
            early_stopping_rounds=50, verbose_eval=0,
        )
        
        xgb_oof[val_idx] = booster.predict(dval)
        xgb_test += booster.predict(dtest) / N_FOLDS
        
        fold_r2 = r2_score(y[val_idx], xgb_oof[val_idx])
        log(f"  XGB Fold {fold_idx+1}: R2={fold_r2:.6f}")
        
        del dtrain, dval, dtest, booster
        gc.collect()
    
    xgb_r2 = r2_score(y, xgb_oof)
    log(f"  XGBoost OOF R2: {xgb_r2:.6f}")
    all_oof["xgboost"] = xgb_oof
    all_test_preds["xgboost"] = xgb_test

except ImportError:
    log("  XGBoost not available, skipping")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SELECT BEST MODEL & BLEND
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 6: Selecting best model and blending...")

# Print all R2 scores
model_scores = {}
for name, oof in all_oof.items():
    r2 = r2_score(y, oof)
    model_scores[name] = r2
    log(f"  {name}: R2={r2:.6f}")

# Find best single model
best_name = max(model_scores, key=model_scores.get)
best_r2 = model_scores[best_name]
log(f"\n  Best single model: {best_name} (R2={best_r2:.6f})")

# Try pairwise blending
log("  Trying blended models...")
model_names = list(all_oof.keys())
best_blend_r2 = best_r2
best_blend_preds = all_test_preds[best_name].copy()
best_blend_oof = all_oof[best_name].copy()

for i, n1 in enumerate(model_names):
    for j, n2 in enumerate(model_names):
        if i >= j:
            continue
        for alpha in np.arange(0.1, 1.0, 0.1):
            blend = all_oof[n1] * alpha + all_oof[n2] * (1 - alpha)
            r2 = r2_score(y, blend)
            if r2 > best_blend_r2:
                best_blend_r2 = r2
                best_blend_preds = all_test_preds[n1] * alpha + all_test_preds[n2] * (1 - alpha)
                best_blend_oof = blend
                log(f"    New best: {n1}*{alpha:.1f} + {n2}*{1-alpha:.1f} = R2={r2:.6f}")

# Try scaling
best_scale = 1.0
for scale in np.arange(0.5, 2.0, 0.05):
    sr2 = r2_score(y, best_blend_oof * scale)
    if sr2 > best_blend_r2:
        best_blend_r2 = sr2
        best_scale = scale
        
if best_scale != 1.0:
    log(f"  Scaling predictions by {best_scale:.2f}: R2 -> {best_blend_r2:.6f}")
    best_blend_preds *= best_scale

# Safety: if model is worse than predicting 0, try blending with 0
zero_r2 = r2_score(y, np.zeros(n_train))
log(f"  Zero-prediction R2: {zero_r2:.6f}")

if best_blend_r2 < zero_r2:
    log("  Model worse than zero, finding optimal shrinkage...")
    best_alpha = 0.0
    best_alpha_r2 = zero_r2
    for alpha in np.arange(0.0, 1.01, 0.01):
        ar2 = r2_score(y, best_blend_oof * alpha * best_scale)
        if ar2 > best_alpha_r2:
            best_alpha_r2 = ar2
            best_alpha = alpha
    log(f"  Best alpha: {best_alpha:.2f}, R2: {best_alpha_r2:.6f}")
    best_blend_preds *= best_alpha
    best_blend_r2 = best_alpha_r2

test_final = best_blend_preds
log(f"\n  FINAL R2: {best_blend_r2:.6f}")
log(f"  Pred stats: mean={test_final.mean():.6f}, std={test_final.std():.6f}")
log(f"  Pred range: [{test_final.min():.6f}, {test_final.max():.6f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. GENERATE SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 7: Generating submission...")

submission = pd.DataFrame({"ID": test_ids, "TARGET": test_final})
submission.to_csv("submission.csv", index=False)
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
log(f"  Saved submission.csv ({len(submission)} rows)")
log(f"  Submission preview:")
log(submission.head(10).to_string())

log("\n" + "="*60)
log("PIPELINE COMPLETE!")
log(f"Final R2: {best_blend_r2:.6f}")
log("="*60)
