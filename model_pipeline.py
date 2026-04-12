"""
Short-Horizon Return Prediction — Competition-Winning Pipeline
================================================================
Advanced multi-model ensemble with:
- Feature selection (top correlated features)
- Target winsorization
- LightGBM with multiple configs (Huber loss, shallow/deep trees)
- Multi-seed blending
- Careful GroupKFold CV
"""

import os
import gc
import json
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
# 1. LOAD DATA & COMPUTE CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 1: Loading data and computing feature correlations...")

# Load target/groups/IDs
train_meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y_full = train_meta["TARGET"].values.astype(np.float32)
groups = train_meta["CV_GROUP"].values.copy()
train_ids = train_meta["ID"].values.copy()
n_train = len(y_full)
del train_meta; gc.collect()

# Get feature names
all_schema_cols = pq.ParquetFile("train.parquet").schema.names
drop_set = {"ID", "CV_GROUP", "TARGET"}
all_feature_names = [c for c in all_schema_cols if c not in drop_set]

# Compute correlation of each feature with target (one column at a time for memory)
log("  Computing correlations with target...")
y_mean = y_full.mean()
y_std = y_full.std()
y_centered = y_full - y_mean

corrs = {}
for i, c in enumerate(all_feature_names):
    col = pd.read_parquet("train.parquet", columns=[c])[c].values.astype(np.float32)
    col_mean = col.mean()
    col_std = col.std()
    if col_std > 0:
        corrs[c] = float(np.dot(y_centered, col - col_mean) / (n_train * y_std * col_std))
    else:
        corrs[c] = 0.0
    del col
    if (i + 1) % 100 == 0:
        log(f"    {i+1}/{len(all_feature_names)}")
        gc.collect()

corrs_sorted = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
log(f"  Top correlations: {corrs_sorted[0][1]:.4f} ({corrs_sorted[0][0]})")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE SELECTION & ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 2: Feature selection and engineering...")

# Select features with |correlation| > threshold - these have signal
CORR_THRESHOLDS = [0.002]  # Use features with at least some correlation
selected_features = [name for name, corr in corrs_sorted if abs(corr) > 0.002]
log(f"  Selected {len(selected_features)} features with |corr| > 0.002")

# Also keep the engineered lag features we'll create
extra_feat_names = []

# ═══════════════════════════════════════════════════════════════════════════════
# 3. BUILD FEATURE MATRICES
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 3: Building feature matrices...")

# Identify lag feature subsets for aggregation
lag_t1 = [c for c in selected_features if "_LagT1" in c]
lag_t2 = [c for c in selected_features if "_LagT2" in c]
lag_t3 = [c for c in selected_features if "_LagT3" in c]

def compute_agg(df_path, cols, l1, l2, l3):
    """Compute aggregate lag features memory-efficiently."""
    n = pq.read_metadata(df_path).num_rows
    feats = {}
    
    for tag, lag_list in [("t1", l1), ("t2", l2), ("t3", l3)]:
        if len(lag_list) == 0:
            feats[f"lag_{tag}_mean"] = np.zeros(n, dtype=np.float32)
            feats[f"lag_{tag}_std"] = np.zeros(n, dtype=np.float32)
            continue
        s = np.zeros(n, np.float32)
        sq = np.zeros(n, np.float32)
        for c in lag_list:
            if c in cols:
                v = pd.read_parquet(df_path, columns=[c])[c].values.astype(np.float32)
                s += v; sq += v * v
                del v
        m = s / max(len(lag_list), 1)
        std = np.sqrt(np.maximum(sq / max(len(lag_list), 1) - m**2, 0))
        feats[f"lag_{tag}_mean"] = m
        feats[f"lag_{tag}_std"] = std
        del s, sq
    
    feats["lag_accel"] = feats["lag_t1_mean"] - feats["lag_t2_mean"]
    feats["lag_jerk"] = feats["lag_accel"] - (feats["lag_t2_mean"] - feats["lag_t3_mean"])
    
    # Cross-lag ratios
    denom = np.abs(feats["lag_t2_mean"]) + 1e-10
    feats["lag_t1_t2_ratio"] = feats["lag_t1_mean"] / denom
    feats["lag_momentum_consistency"] = feats["lag_t1_std"] / (feats["lag_t2_std"] + 1e-10)
    
    return feats

extra_feat_names = ["lag_t1_mean", "lag_t1_std", "lag_t2_mean", "lag_t2_std",
                    "lag_t3_mean", "lag_t3_std", "lag_accel", "lag_jerk",
                    "lag_t1_t2_ratio", "lag_momentum_consistency"]

all_lag_t1 = [c for c in all_feature_names if "_LagT1" in c]
all_lag_t2 = [c for c in all_feature_names if "_LagT2" in c]
all_lag_t3 = [c for c in all_feature_names if "_LagT3" in c]

# Build train matrix
log("  Building train matrix...")
train_agg = compute_agg("train.parquet", all_feature_names, all_lag_t1, all_lag_t2, all_lag_t3)
n_selected = len(selected_features)
n_extra = len(extra_feat_names)
n_total = n_selected + n_extra

X_train = np.empty((n_train, n_total), dtype=np.float32)
for i, c in enumerate(selected_features):
    X_train[:, i] = pd.read_parquet("train.parquet", columns=[c])[c].values.astype(np.float32)
    if (i + 1) % 100 == 0:
        log(f"    {i+1}/{n_selected}")
        gc.collect()
for j, name in enumerate(extra_feat_names):
    X_train[:, n_selected + j] = train_agg[name]
del train_agg; gc.collect()

feature_names = selected_features + extra_feat_names
log(f"  X_train: {X_train.shape}")

# Build test matrix
log("  Building test matrix...")
test_meta = pd.read_parquet("test.parquet", columns=["ID"])
test_ids = test_meta["ID"].values.copy()
n_test = len(test_ids)
del test_meta; gc.collect()

# Check which selected features exist in test
test_cols = pq.ParquetFile("test.parquet").schema.names
missing = [c for c in selected_features if c not in test_cols]
if missing:
    log(f"  WARNING: {len(missing)} features missing in test, replacing with 0")

test_agg = compute_agg("test.parquet", test_cols, all_lag_t1, all_lag_t2, all_lag_t3)
X_test = np.empty((n_test, n_total), dtype=np.float32)
for i, c in enumerate(selected_features):
    if c in test_cols:
        X_test[:, i] = pd.read_parquet("test.parquet", columns=[c])[c].values.astype(np.float32)
    else:
        X_test[:, i] = 0.0
    if (i + 1) % 100 == 0:
        gc.collect()
for j, name in enumerate(extra_feat_names):
    X_test[:, n_selected + j] = test_agg[name]
del test_agg; gc.collect()
log(f"  X_test: {X_test.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TARGET PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 4: Target preprocessing (winsorization)...")

# Clip extreme outliers (P1/P99) for training stability
lo_clip = np.percentile(y_full, 1)
hi_clip = np.percentile(y_full, 99)
y_clipped = np.clip(y_full, lo_clip, hi_clip)
log(f"  Clipped target to [{lo_clip:.4f}, {hi_clip:.4f}]")
log(f"  Clipped std: {y_clipped.std():.6f} vs original: {y_full.std():.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MULTI-CONFIG LIGHTGBM ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 5: Training multi-config LightGBM ensemble...")

# Define diverse model configurations
CONFIGS = [
    {
        "name": "lgb_huber_shallow",
        "params": {
            "objective": "huber",
            "alpha": 0.5,  # Huber delta - robust to outliers
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 200,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "min_gain_to_split": 0.01,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 10000,
        "early_stopping": 300,
        "use_clipped": True,
        "seeds": [42, 123, 456],
        "weight": 0.35,
    },
    {
        "name": "lgb_mse_deep",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.005,
            "num_leaves": 63,
            "max_depth": 7,
            "min_child_samples": 300,
            "feature_fraction": 0.4,
            "bagging_fraction": 0.6,
            "bagging_freq": 1,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
            "min_gain_to_split": 0.005,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 10000,
        "early_stopping": 300,
        "use_clipped": True,
        "seeds": [42, 789, 101],
        "weight": 0.35,
    },
    {
        "name": "lgb_fair_regularized",
        "params": {
            "objective": "fair",
            "fair_c": 1.0,  # Fair loss - very robust
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 500,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "reg_alpha": 5.0,
            "reg_lambda": 20.0,
            "min_gain_to_split": 0.02,
            "verbose": -1,
            "n_jobs": -1,
        },
        "num_boost_round": 10000,
        "early_stopping": 300,
        "use_clipped": True,
        "seeds": [42, 202, 303],
        "weight": 0.30,
    },
]

gkf = GroupKFold(n_splits=N_FOLDS)
all_oof = {}
all_test_preds = {}
all_fold_details = []
all_feature_importance = np.zeros(n_total)

for cfg in CONFIGS:
    config_name = cfg["name"]
    log(f"\n  === Config: {config_name} (weight={cfg['weight']}) ===")
    
    y_target = y_clipped if cfg["use_clipped"] else y_full
    
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
                    lgb.log_evaluation(period=0),
                ],
            )
            
            seed_oof[val_idx] = booster.predict(X_train[val_idx])
            seed_test += booster.predict(X_test) / N_FOLDS
            
            if seed_idx == 0:
                all_feature_importance += booster.feature_importance(importance_type="gain")
            
            del dtrain, dval, booster
            gc.collect()
        
        # Evaluate this seed (on ORIGINAL target, not clipped)
        seed_r2 = r2_score(y_full, seed_oof)
        log(f"      Seed R2 (vs original target): {seed_r2:.6f}")
        
        config_oof += seed_oof / n_seeds
        config_test += seed_test / n_seeds
    
    config_r2 = r2_score(y_full, config_oof)
    log(f"    Config {config_name} OOF R2: {config_r2:.6f}")
    
    all_oof[config_name] = config_oof
    all_test_preds[config_name] = config_test
    
    all_fold_details.append({
        "config": config_name,
        "r2": round(float(config_r2), 6),
        "weight": cfg["weight"],
        "seeds": cfg["seeds"],
        "params": {k: str(v) for k, v in cfg["params"].items()},
    })

# ═══════════════════════════════════════════════════════════════════════════════
# 6. WEIGHTED ENSEMBLE BLENDING
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 6: Weighted ensemble blending...")

# Weighted average
total_weight = sum(cfg["weight"] for cfg in CONFIGS)
oof_ensemble = np.zeros(n_train, dtype=np.float64)
test_ensemble = np.zeros(n_test, dtype=np.float64)

for cfg in CONFIGS:
    name = cfg["name"]
    w = cfg["weight"] / total_weight
    oof_ensemble += all_oof[name] * w
    test_ensemble += all_test_preds[name] * w

ensemble_r2 = r2_score(y_full, oof_ensemble)
log(f"  Ensemble OOF R2: {ensemble_r2:.6f}")

# Try optimizing weights with brute force search
log("  Searching for optimal weights...")
best_r2 = ensemble_r2
best_weights = [cfg["weight"] for cfg in CONFIGS]
config_names = [cfg["name"] for cfg in CONFIGS]

for w0 in np.arange(0.1, 0.9, 0.05):
    for w1 in np.arange(0.1, 0.9 - w0, 0.05):
        w2 = 1.0 - w0 - w1
        if w2 < 0.05:
            continue
        trial_oof = (all_oof[config_names[0]] * w0 + 
                     all_oof[config_names[1]] * w1 + 
                     all_oof[config_names[2]] * w2)
        trial_r2 = r2_score(y_full, trial_oof)
        if trial_r2 > best_r2:
            best_r2 = trial_r2
            best_weights = [w0, w1, w2]

log(f"  Optimal weights: {best_weights}")
log(f"  Optimal R2: {best_r2:.6f}")

# Apply optimal weights
test_final = np.zeros(n_test, dtype=np.float64)
oof_final = np.zeros(n_train, dtype=np.float64)
for i, name in enumerate(config_names):
    test_final += all_test_preds[name] * best_weights[i]
    oof_final += all_oof[name] * best_weights[i]

final_r2 = r2_score(y_full, oof_final)
log(f"  Final OOF R2: {final_r2:.6f}")

# Also check: does predicting 0 for all beat us?
zero_r2 = r2_score(y_full, np.zeros(n_train))
log(f"  Baseline (predict 0) R2: {zero_r2:.6f}")

# If our model is worse than predicting 0, blend with 0
if final_r2 < zero_r2:
    log("  WARNING: Model worse than zero prediction, blending with zeros...")
    best_blend_r2 = final_r2
    best_blend_alpha = 1.0
    for alpha in np.arange(0, 1.01, 0.01):
        blended = oof_final * alpha
        br2 = r2_score(y_full, blended)
        if br2 > best_blend_r2:
            best_blend_r2 = br2
            best_blend_alpha = alpha
    log(f"  Best blend alpha: {best_blend_alpha:.2f}, R2: {best_blend_r2:.6f}")
    test_final = test_final * best_blend_alpha
    oof_final = oof_final * best_blend_alpha
    final_r2 = best_blend_r2
else:
    # Still try scaling to see if it helps
    best_scale_r2 = final_r2
    best_scale = 1.0
    for scale in np.arange(0.5, 1.5, 0.01):
        sr2 = r2_score(y_full, oof_final * scale)
        if sr2 > best_scale_r2:
            best_scale_r2 = sr2
            best_scale = scale
    if best_scale != 1.0:
        log(f"  Scaling predictions by {best_scale:.2f}: R2 {final_r2:.6f} -> {best_scale_r2:.6f}")
        test_final *= best_scale
        oof_final *= best_scale
        final_r2 = best_scale_r2

log(f"\n  FINAL R2: {final_r2:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. GENERATE SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 7: Generating submission...")
submission = pd.DataFrame({"ID": test_ids, "TARGET": test_final})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
log(f"  Saved submission.csv ({len(submission)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. SAVE DASHBOARD ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 8: Saving dashboard artifacts...")

from sklearn.metrics import mean_squared_error, mean_absolute_error
overall_rmse = float(np.sqrt(mean_squared_error(y_full, oof_final)))
overall_mae = float(mean_absolute_error(y_full, oof_final))

# Per-fold metrics for best config
fold_details_cv = []
for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_full, groups)):
    fold_r2 = r2_score(y_full[val_idx], oof_final[val_idx])
    fold_rmse = float(np.sqrt(mean_squared_error(y_full[val_idx], oof_final[val_idx])))
    fold_mae = float(mean_absolute_error(y_full[val_idx], oof_final[val_idx]))
    fold_details_cv.append({
        "fold": fold_idx + 1,
        "train_size": int(len(tr_idx)),
        "val_size": int(len(val_idx)),
        "r2": round(float(fold_r2), 6),
        "rmse": round(fold_rmse, 6),
        "mae": round(fold_mae, 6),
    })

cv_results = {
    "overall_r2": round(float(final_r2), 6),
    "overall_rmse": round(overall_rmse, 6),
    "overall_mae": round(overall_mae, 6),
    "mean_r2": round(float(np.mean([f["r2"] for f in fold_details_cv])), 6),
    "std_r2": round(float(np.std([f["r2"] for f in fold_details_cv])), 6),
    "n_folds": N_FOLDS,
    "n_train": n_train,
    "n_test": n_test,
    "n_features": n_total,
    "folds": fold_details_cv,
    "ensemble_configs": all_fold_details,
    "optimal_weights": best_weights,
}
with open(os.path.join(OUTPUT_DIR, "cv_results.json"), "w") as f:
    json.dump(cv_results, f, indent=2)

# Feature importance
fi = all_feature_importance / (N_FOLDS * len(CONFIGS[0]["seeds"]))
fi_order = np.argsort(fi)[::-1][:50]
feature_importance = {
    "features": [feature_names[i] for i in fi_order],
    "gain": [round(float(fi[i]), 2) for i in fi_order],
    "split": [round(float(fi[i]), 2) for i in fi_order],  
}
with open(os.path.join(OUTPUT_DIR, "feature_importance.json"), "w") as f:
    json.dump(feature_importance, f, indent=2)

# Prediction analysis
def make_histogram(arr, n_bins=50):
    counts, bin_edges = np.histogram(arr, bins=n_bins)
    return {"counts": counts.tolist(), "bin_edges": [round(float(b), 6) for b in bin_edges]}

residuals = y_full - oof_final.astype(np.float32)
np.random.seed(SEED)
sample_idx = np.random.choice(n_train, size=min(5000, n_train), replace=False)

predictions_analysis = {
    "actual_hist": make_histogram(y_full, 80),
    "pred_hist": make_histogram(oof_final, 80),
    "residual_hist": make_histogram(residuals, 80),
    "scatter": {
        "actual": [round(float(y_full[i]), 6) for i in sample_idx],
        "predicted": [round(float(oof_final[i]), 6) for i in sample_idx],
    },
    "test_pred_hist": make_histogram(test_final, 80),
    "actual_stats": {
        "mean": round(float(np.mean(y_full)), 6),
        "std": round(float(np.std(y_full)), 6),
        "min": round(float(np.min(y_full)), 6),
        "max": round(float(np.max(y_full)), 6),
        "median": round(float(np.median(y_full)), 6),
    },
    "pred_stats": {
        "mean": round(float(np.mean(test_final)), 6),
        "std": round(float(np.std(test_final)), 6),
        "min": round(float(np.min(test_final)), 6),
        "max": round(float(np.max(test_final)), 6),
        "median": round(float(np.median(test_final)), 6),
    },
}
with open(os.path.join(OUTPUT_DIR, "predictions_analysis.json"), "w") as f:
    json.dump(predictions_analysis, f, indent=2)

model_config = {
    "model_type": "LightGBM Ensemble",
    "n_configs": len(CONFIGS),
    "configs": [c["name"] for c in CONFIGS],
    "optimal_weights": [round(w, 3) for w in best_weights],
    "cv_strategy": "GroupKFold",
    "n_folds": N_FOLDS,
    "feature_selection": f"Top {n_selected} by |correlation| > 0.002 + {n_extra} engineered",
    "target_preprocessing": f"Winsorized P1/P99 [{lo_clip:.4f}, {hi_clip:.4f}]",
    "total_features": n_total,
    "loss_functions": ["Huber", "MSE", "Fair"],
    "multi_seed": True,
    "seeds_per_config": len(CONFIGS[0]["seeds"]),
    "total_models": sum(len(c["seeds"]) * N_FOLDS for c in CONFIGS),
}
with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(model_config, f, indent=2)

log("")
log("Pipeline complete! All artifacts saved to output/")
log(f"   Final R2 score: {final_r2:.6f}")
log(f"   Total models trained: {model_config['total_models']}")
