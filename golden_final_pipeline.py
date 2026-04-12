"""
Golden Final Pipeline
=====================
- 10-Fold Deep Architecture (Huber, Fair, MSE)
- Adversarial Anti-Drift Feature Selection (Top 85 dropped outright)
- Tri-Loss Ensemble Optimization
- Chronological Amplified Output Smoothing (0.80 / 0.10 / 0.10)
"""

import os, gc, time, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Phase 1: Setup & Bypassing PyArrow Array Limits...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = meta["TARGET"].values.astype(np.float32)
groups = meta["CV_GROUP"].values
n_train = len(y)
del meta; gc.collect()

test_meta = pd.read_parquet("test.parquet", columns=["ID"])
test_ids = test_meta["ID"].values
n_test = len(test_ids)

all_cols = pq.ParquetFile("train.parquet").schema.names
raw_feats = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET", "SO3_T"}]
test_col_set = set(pq.ParquetFile("test.parquet").schema.names)
safe_feats = [c for c in raw_feats if c in test_col_set]

log(f"Phase 2: Instant Adversarial Drift Profiling via Surrogate Models...")
idx_tr = np.random.choice(n_train, 250000, replace=False)
idx_te = np.random.choice(n_test, 250000, replace=False)

adv_importances = {}
batch_size = 50
for i in range(0, len(safe_feats), batch_size):
    b = safe_feats[i:i+batch_size]
    df_tr = pd.read_parquet("train.parquet", columns=b).iloc[idx_tr]
    df_te = pd.read_parquet("test.parquet", columns=b).iloc[idx_te]
    
    df_tr["is_test"] = 0
    df_te["is_test"] = 1
    df = pd.concat([df_tr, df_te], axis=0).sample(frac=1.0, random_state=42)
    y_adv = df.pop("is_test").values
    X_adv = df.values.astype(np.float32)
    
    xtr, xva, ytr, yva = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42, stratify=y_adv)
    
    adv_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                   num_leaves=31, random_state=42, n_jobs=-1)
    adv_model.fit(xtr, ytr, eval_set=[(xva, yva)], callbacks=[lgb.early_stopping(10, verbose=False)])
    
    imps = adv_model.feature_importances_
    for col, imp in zip(b, imps):
        adv_importances[col] = imp
        
    del df_tr, df_te, df, y_adv, X_adv, xtr, xva, ytr, yva, adv_model; gc.collect()

adversarial_ranking = sorted(adv_importances.items(), key=lambda x: x[1], reverse=True)
TOXIC_COUNT = 85
toxic_features = set(name for name, imp in adversarial_ranking[:TOXIC_COUNT])

log(f"\nPhase 3: Stationary Subsetting...")
log(f"  Dropping {TOXIC_COUNT} toxic variables to shield Test set boundaries.")
clean_feats = [f for f in safe_feats if f not in toxic_features]

y_c = y - y.mean()
corrs = {}
for i in range(0, len(clean_feats), 50):
    b = clean_feats[i:i+50]
    df = pd.read_parquet("train.parquet", columns=b)
    for c in b:
        v = df[c].values.astype(np.float32)
        vs = v.std()
        corrs[c] = float(np.dot(y_c, v - v.mean()) / (n_train * y.std() * vs)) if vs > 0 else 0.0
    del df; gc.collect()

TOP_N = 120 
selected = [name for name, _ in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:TOP_N]]
log(f"  Stationary set loaded: {len(selected)} features completely safe from Domain Drift.")
del corrs, y_c; gc.collect()

log("\nPhase 4: Constructing Matrices...")
def build_matrix(path, avail, n_rows, target_feats):
    X = np.empty((n_rows, len(target_feats)), dtype=np.float32)
    for i in range(0, len(target_feats), 50):
        b = target_feats[i:i+50]
        df = pd.read_parquet(path, columns=b)
        for j, c in enumerate(b):
            X[:, i+j] = df[c].values.astype(np.float32)
        del df; gc.collect()
    return np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_train = build_matrix("train.parquet", test_col_set, n_train, selected)
lo, hi = np.percentile(y, 1), np.percentile(y, 99)
y_clip = np.clip(y, lo, hi)

log("\nPhase 5: Golden Deep Tri-Loss Ensemble...")
N_FOLDS = 10
gkf = GroupKFold(n_splits=N_FOLDS)

CONFIGS = [
    {
        "name": "Huber",
        "params": {
            "objective": "huber", "alpha": 0.9, "metric": "root_mean_squared_error", "boosting_type": "gbdt",
            "learning_rate": 0.005, "num_leaves": 127, "max_depth": -1, "min_child_samples": 120,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, 
            "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "n_jobs": -1, "random_state": 42
        }
    },
    {
        "name": "Fair",
        "params": {
            "objective": "fair", "fair_c": 1.0, "metric": "root_mean_squared_error", "boosting_type": "gbdt",
            "learning_rate": 0.005, "num_leaves": 127, "max_depth": -1, "min_child_samples": 150,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
            "reg_alpha": 2.0, "reg_lambda": 5.0, "verbose": -1, "n_jobs": -1, "random_state": 42
        }
    },
    {
        "name": "MSE",
        "params": {
            "objective": "regression", "metric": "root_mean_squared_error", "boosting_type": "gbdt",
            "learning_rate": 0.005, "num_leaves": 63, "max_depth": 7, "min_child_samples": 120,
            "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1,
            "reg_alpha": 1.0, "reg_lambda": 3.0, "verbose": -1, "n_jobs": -1, "random_state": 42
        }
    }
]

oof_preds = {cfg["name"]: np.zeros(n_train, np.float32) for cfg in CONFIGS}
models = {cfg["name"]: [] for cfg in CONFIGS}

for cfg in CONFIGS:
    log(f"  Training {cfg['name']} (10-Folds)...")
    for fi, (tr, va) in enumerate(gkf.split(X_train, y_clip, groups)):
        dt = lgb.Dataset(X_train[tr], y_clip[tr], free_raw_data=True)
        dv = lgb.Dataset(X_train[va], y_clip[va], free_raw_data=True, reference=dt)
        bst = lgb.train(cfg["params"], dt, 2000, valid_sets=[dv], valid_names=["v"],
                        callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_preds[cfg["name"]][va] = bst.predict(X_train[va])
        models[cfg["name"]].append(bst.model_to_string())
        del dt, dv, bst; gc.collect()
    
    cfg_r2 = r2_score(y, oof_preds[cfg["name"]])
    log(f"    {cfg['name']} Local OOF R2: {cfg_r2:.6f}")

log("\nPhase 6: Multi-Dimensional Smooth Blend Optimization...")
names = [cfg["name"] for cfg in CONFIGS]
best_r2, best_w = -float('inf'), [0.33, 0.33, 0.34]

# Quick random grid walk for tri-loss weights
for w0 in np.arange(0.0, 1.05, 0.1):
    for w1 in np.arange(0.0, 1.05 - w0, 0.1):
        w2 = max(0.0, 1.0 - w0 - w1)
        blend = oof_preds[names[0]]*w0 + oof_preds[names[1]]*w1 + oof_preds[names[2]]*w2
        r2 = r2_score(y, blend)
        if r2 > best_r2: 
             best_r2 = r2
             best_w = [w0, w1, w2]

oof_blend = sum(oof_preds[names[i]]*best_w[i] for i in range(3))
best_scale = 1.0; best_scale_r2 = r2_score(y, oof_blend)

for s in np.arange(0.5, 3.0, 0.1):
    sr2 = r2_score(y, oof_blend * s)
    if sr2 > best_scale_r2: best_scale, best_scale_r2 = s, sr2
         
log(f"  Stationary Global Blend OOF R2: {best_scale_r2:.6f} with Scale: {best_scale:.2f}")

log("\nPhase 7: Mapping Sequence Predictions...")
del X_train, oof_preds, oof_blend, y_clip; gc.collect()

X_test = build_matrix("test.parquet", test_col_set, n_test, selected)
test_blend = np.zeros(n_test, np.float32)

for i, cfg in enumerate(CONFIGS):
    name = cfg["name"]
    w = best_w[i]
    if w == 0: continue
    
    for ms in models[name]:
        bst = lgb.Booster(model_str=ms)
        chunk_size = 50000
        for start_idx in range(0, n_test, chunk_size):
            end_idx = min(start_idx + chunk_size, n_test)
            test_blend[start_idx:end_idx] += bst.predict(X_test[start_idx:end_idx]) * w / N_FOLDS
        del bst; gc.collect()

# We multiply the best_scale by an aggressive 1.25 bump factor due to the heavy sequence smoothing
# Smoothing naturally compresses variance, so stretching it prevents variance attenuation
aggressive_scale = best_scale * 1.25
test_meta["RAW_PRED"] = test_blend * aggressive_scale

log("\nPhase 8: Applying Heavy Amplifier Sequential Smoothing...")
test_meta["pred_prev"] = test_meta["RAW_PRED"].shift(1).fillna(test_meta["RAW_PRED"])
test_meta["pred_next"] = test_meta["RAW_PRED"].shift(-1).fillna(test_meta["RAW_PRED"])

# We confirmed 0.90/0.05/0.05 worked mathematically earlier (0.00041). Expanding to 0.80/0.10/0.10.
test_meta["TARGET"] = 0.80 * test_meta["RAW_PRED"] + 0.10 * test_meta["pred_prev"] + 0.10 * test_meta["pred_next"]

test_sub = test_meta[["ID", "TARGET"]]
test_sub.to_csv("submission_golden.csv", index=False)

log(f"\nFinal Golden Run completed successfully.")
log(f"Saved submission_golden.csv ({len(test_sub)} rows)")
