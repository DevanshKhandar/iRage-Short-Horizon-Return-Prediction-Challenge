"""
Short-Horizon Return Prediction — ENHANCED Pipeline V3 (Memory Optimized)
===========================================================================
Key improvements:
1. ALL features (445 raw + 14 engineered = 459)
2. Minimal regularization
3. Expressive trees (127-255 leaves)
4. Raw target (no winsorization)
5. 3 diverse configs × 2 seeds = 30 models total
6. Memory-optimized: sequential fold processing, aggressive GC
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
log("=" * 70)
log("ENHANCED PIPELINE V3 (Memory Optimized)")
log("=" * 70)
log("Step 1: Loading data...")

train_meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y_full = train_meta["TARGET"].values.astype(np.float32)
groups = train_meta["CV_GROUP"].values.copy()
train_ids = train_meta["ID"].values.copy()
n_train = len(y_full)
del train_meta; gc.collect()

log(f"  Train: {n_train}, Target mean={y_full.mean():.6f}, std={y_full.std():.6f}")

# Get ALL feature names
all_schema_cols = pq.ParquetFile("train.parquet").schema.names
drop_set = {"ID", "CV_GROUP", "TARGET"}
raw_feature_names = [c for c in all_schema_cols if c not in drop_set]
n_raw = len(raw_feature_names)
log(f"  Raw features: {n_raw}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD FEATURE MATRICES (MEMORY EFFICIENT)
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 2: Building feature matrices...")

CHUNK = 50

def load_features(path, feat_names, schema_cols):
    n = pq.read_metadata(path).num_rows
    nf = len(feat_names)
    X = np.empty((n, nf), dtype=np.float32)
    
    for start in range(0, nf, CHUNK):
        end = min(start + CHUNK, nf)
        cols = feat_names[start:end]
        avail = [c for c in cols if c in schema_cols]
        miss = [c for c in cols if c not in schema_cols]
        
        if avail:
            df = pd.read_parquet(path, columns=avail)
            for c in avail:
                X[:, feat_names.index(c)] = df[c].values.astype(np.float32)
            del df
        for c in miss:
            X[:, feat_names.index(c)] = 0.0
        
        if end % 200 == 0 or end == nf:
            log(f"    {end}/{nf}")
            gc.collect()
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

# Load train
log("  Loading train features...")
train_schema = pq.ParquetFile("train.parquet").schema.names
X_train = load_features("train.parquet", raw_feature_names, train_schema)
log(f"  X_train: {X_train.shape}")

# Engineer features on the raw matrix
log("  Engineering features...")
feat_idx = {name: i for i, name in enumerate(raw_feature_names)}

eng_cols = []
eng_names = []

# Lag aggregate features
lag_t1 = [feat_idx[c] for c in raw_feature_names if "_LagT1" in c]
lag_t2 = [feat_idx[c] for c in raw_feature_names if "_LagT2" in c]
lag_t3 = [feat_idx[c] for c in raw_feature_names if "_LagT3" in c]

if lag_t1:
    d = X_train[:, lag_t1]
    eng_cols.append(d.mean(axis=1)); eng_names.append("eng_t1_mean")
    eng_cols.append(d.std(axis=1));  eng_names.append("eng_t1_std")
    del d
if lag_t2:
    d = X_train[:, lag_t2]
    eng_cols.append(d.mean(axis=1)); eng_names.append("eng_t2_mean")
    eng_cols.append(d.std(axis=1));  eng_names.append("eng_t2_std")
    del d
if lag_t3:
    d = X_train[:, lag_t3]
    eng_cols.append(d.mean(axis=1)); eng_names.append("eng_t3_mean")
    eng_cols.append(d.std(axis=1));  eng_names.append("eng_t3_std")
    del d

# Momentum/acceleration
if len(eng_cols) >= 4:  # t1_mean, t1_std, t2_mean, t2_std
    eng_cols.append(eng_cols[0] - eng_cols[2])  # accel
    eng_names.append("eng_accel")
    if len(eng_cols) >= 6:  # have t3
        eng_cols.append(eng_cols[-1] - (eng_cols[2] - eng_cols[4]))  # jerk
        eng_names.append("eng_jerk")

# Price diffs
if "Price" in feat_idx:
    p = X_train[:, feat_idx["Price"]]
    for tag in ["Price_LagT1", "Price_LagT2", "Price_LagT3"]:
        if tag in feat_idx:
            eng_cols.append(p - X_train[:, feat_idx[tag]])
            eng_names.append(f"eng_{tag}_diff")

n_eng = len(eng_cols)
log(f"  Engineered {n_eng} features")

# Combine into final train matrix
feature_names = raw_feature_names + eng_names
n_features = len(feature_names)

X_train_full = np.empty((n_train, n_features), dtype=np.float32)
X_train_full[:, :n_raw] = X_train
for j in range(n_eng):
    X_train_full[:, n_raw + j] = eng_cols[j].astype(np.float32)
del X_train, eng_cols
gc.collect()

log(f"  X_train_full: {X_train_full.shape}")

# Load test
log("  Loading test features...")
test_meta = pd.read_parquet("test.parquet", columns=["ID"])
test_ids = test_meta["ID"].values.copy()
n_test = len(test_ids)
del test_meta; gc.collect()

test_schema = pq.ParquetFile("test.parquet").schema.names
X_test_raw = load_features("test.parquet", raw_feature_names, test_schema)

# Engineer test features
log("  Engineering test features...")
test_eng_cols = []

lag_t1_test = [feat_idx[c] for c in raw_feature_names if "_LagT1" in c]
lag_t2_test = [feat_idx[c] for c in raw_feature_names if "_LagT2" in c]
lag_t3_test = [feat_idx[c] for c in raw_feature_names if "_LagT3" in c]

if lag_t1_test:
    d = X_test_raw[:, lag_t1_test]
    test_eng_cols.append(d.mean(axis=1)); test_eng_cols.append(d.std(axis=1)); del d
if lag_t2_test:
    d = X_test_raw[:, lag_t2_test]
    test_eng_cols.append(d.mean(axis=1)); test_eng_cols.append(d.std(axis=1)); del d
if lag_t3_test:
    d = X_test_raw[:, lag_t3_test]
    test_eng_cols.append(d.mean(axis=1)); test_eng_cols.append(d.std(axis=1)); del d

if len(test_eng_cols) >= 4:
    test_eng_cols.append(test_eng_cols[0] - test_eng_cols[2])
    if len(test_eng_cols) >= 6:
        test_eng_cols.append(test_eng_cols[-1] - (test_eng_cols[2] - test_eng_cols[4]))

if "Price" in feat_idx:
    p = X_test_raw[:, feat_idx["Price"]]
    for tag in ["Price_LagT1", "Price_LagT2", "Price_LagT3"]:
        if tag in feat_idx:
            test_eng_cols.append(p - X_test_raw[:, feat_idx[tag]])

X_test_full = np.empty((n_test, n_features), dtype=np.float32)
X_test_full[:, :n_raw] = X_test_raw
for j in range(len(test_eng_cols)):
    X_test_full[:, n_raw + j] = test_eng_cols[j].astype(np.float32)
del X_test_raw, test_eng_cols
gc.collect()

log(f"  X_test_full: {X_test_full.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN LIGHTGBM — FOLD-BY-FOLD TO SAVE MEMORY
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 3: Training LightGBM (fold-by-fold, memory optimized)...")

CONFIGS = [
    {
        "name": "lgb_mse_main",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "max_depth": -1,
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
        "seeds": [42, 123],
        "weight": 0.40,
    },
    {
        "name": "lgb_huber",
        "params": {
            "objective": "huber",
            "alpha": 0.9,
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
        "seeds": [42, 789],
        "weight": 0.35,
    },
    {
        "name": "lgb_deep",
        "params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 255,
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
        "seeds": [42, 202],
        "weight": 0.25,
    },
]

gkf = GroupKFold(n_splits=N_FOLDS)
all_oof = {}
all_test_preds = {}
all_feature_importance = np.zeros(n_features)

# Pre-compute fold indices
fold_indices = list(gkf.split(X_train_full, y_full, groups))

for cfg in CONFIGS:
    config_name = cfg["name"]
    log(f"\n  === Config: {config_name} (weight={cfg['weight']}) ===")
    
    config_oof = np.zeros(n_train, dtype=np.float64)
    config_test = np.zeros(n_test, dtype=np.float64)
    n_seeds = len(cfg["seeds"])
    
    for seed_idx, seed in enumerate(cfg["seeds"]):
        log(f"    Seed {seed_idx+1}/{n_seeds} (seed={seed})")
        
        params = cfg["params"].copy()
        params["random_state"] = seed
        
        seed_oof = np.zeros(n_train, dtype=np.float64)
        seed_test = np.zeros(n_test, dtype=np.float64)
        
        for fold_idx, (tr_idx, val_idx) in enumerate(fold_indices):
            # Create LightGBM datasets — let it manage memory
            dtrain = lgb.Dataset(
                X_train_full[tr_idx], label=y_full[tr_idx],
                feature_name=feature_names, free_raw_data=True
            )
            dval = lgb.Dataset(
                X_train_full[val_idx], label=y_full[val_idx],
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
            
            val_pred = booster.predict(X_train_full[val_idx])
            seed_oof[val_idx] = val_pred
            seed_test += booster.predict(X_test_full) / N_FOLDS
            
            fold_r2 = r2_score(y_full[val_idx], val_pred)
            log(f"      Fold {fold_idx+1}: R2={fold_r2:.6f} (iters={booster.best_iteration})")
            
            if seed_idx == 0:
                all_feature_importance += booster.feature_importance(importance_type="gain")
            
            del dtrain, dval, booster, val_pred
            gc.collect()
        
        seed_r2 = r2_score(y_full, seed_oof)
        log(f"      Seed R2: {seed_r2:.6f}, pred_std={seed_oof.std():.6f}")
        
        config_oof += seed_oof / n_seeds
        config_test += seed_test / n_seeds
        del seed_oof, seed_test
        gc.collect()
    
    config_r2 = r2_score(y_full, config_oof)
    log(f"    Config {config_name} OOF R2: {config_r2:.6f}")
    
    all_oof[config_name] = config_oof
    all_test_preds[config_name] = config_test

# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE BLENDING
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 4: Optimal ensemble blending...")

config_names = [cfg["name"] for cfg in CONFIGS]

for name in config_names:
    log(f"  Solo {name}: R2={r2_score(y_full, all_oof[name]):.6f}")

# Grid search weights
best_r2 = -999
best_weights = [cfg["weight"] for cfg in CONFIGS]

for w0 in np.arange(0.0, 1.01, 0.05):
    for w1 in np.arange(0.0, 1.01 - w0, 0.05):
        w2 = 1.0 - w0 - w1
        if w2 < 0:
            continue
        trial = (all_oof[config_names[0]] * w0 +
                 all_oof[config_names[1]] * w1 +
                 all_oof[config_names[2]] * w2)
        tr2 = r2_score(y_full, trial)
        if tr2 > best_r2:
            best_r2 = tr2
            best_weights = [w0, w1, w2]

log(f"  Optimal weights: {[round(w, 3) for w in best_weights]}")
log(f"  Ensemble R2: {best_r2:.6f}")

# Apply
oof_final = np.zeros(n_train, dtype=np.float64)
test_final = np.zeros(n_test, dtype=np.float64)
for i, name in enumerate(config_names):
    oof_final += all_oof[name] * best_weights[i]
    test_final += all_test_preds[name] * best_weights[i]

# Check solo beats ensemble
for name in config_names:
    sr2 = r2_score(y_full, all_oof[name])
    if sr2 > best_r2:
        log(f"  USING solo {name} (R2={sr2:.6f})")
        oof_final = all_oof[name].copy()
        test_final = all_test_preds[name].copy()
        best_r2 = sr2

# ═══════════════════════════════════════════════════════════════════════════════
# 5. POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
log("\nStep 5: Post-processing...")

# Optimal scaling
best_scale = 1.0
best_sr2 = best_r2
for s in np.arange(0.5, 2.0, 0.01):
    sr2 = r2_score(y_full, oof_final * s)
    if sr2 > best_sr2:
        best_sr2 = sr2
        best_scale = s

if best_scale != 1.0:
    log(f"  Scale {best_scale:.2f}: R2 {best_r2:.6f} -> {best_sr2:.6f}")
    test_final *= best_scale
    oof_final *= best_scale
    best_r2 = best_sr2

# Zero baseline check
zero_r2 = r2_score(y_full, np.zeros(n_train))
log(f"  Zero baseline R2: {zero_r2:.6f}")

if best_r2 < zero_r2:
    log("  WARNING: Below zero baseline, alpha blending...")
    ba, bar2 = 1.0, best_r2
    for a in np.arange(0, 1.01, 0.01):
        ar2 = r2_score(y_full, oof_final * a)
        if ar2 > bar2:
            bar2 = ar2; ba = a
    log(f"  Alpha={ba:.2f}, R2={bar2:.6f}")
    test_final *= ba; oof_final *= ba; best_r2 = bar2

final_r2 = r2_score(y_full, oof_final)
log(f"\n  *** FINAL R2: {final_r2:.6f} ***")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
log("Step 6: Saving submission...")

sub = pd.DataFrame({"ID": test_ids, "TARGET": test_final})
sub.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
sub.to_csv("submission.csv", index=False)
log(f"  Saved ({len(sub)} rows)")
log(f"  mean={test_final.mean():.6f}, std={test_final.std():.6f}")
log(f"  min={test_final.min():.6f}, max={test_final.max():.6f}")

# Top features
fi = all_feature_importance / (N_FOLDS * len(CONFIGS[0]["seeds"]))
fi_order = np.argsort(fi)[::-1][:20]
log("\n  Top 20 Features:")
for r, idx in enumerate(fi_order):
    log(f"    {r+1:2d}. {feature_names[idx]:40s}: {fi[idx]:.1f}")

log(f"\nDONE! Final R2: {final_r2:.6f}")
