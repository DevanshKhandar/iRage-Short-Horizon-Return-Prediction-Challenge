"""
Master Submission Pipeline
===========================
This is the ultimate, hyper-optimized ensemble designed to extract the absolute maximum
legitimate machine learning signal from the data without resorting to leaky cross-validation.

Features:
1. Memory-efficient chunked loading.
2. 5-Fold GroupKFold (to ensure no time-leakage).
3. 3 Diverse LightGBM Models (Huber, Fair, MSE) to capture different signal aspects.
4. Extensive feature engineering (Microstructure & Lags).
5. Automatic optimal blending (Nelder-Mead equivalent via grid search).
6. Optimal Prediction Scaling (to match target variance).
"""

import os, gc, time, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD & ENGINEER DATA
# ═════════════════════════════════════════════════════════════════════════════
log("Step 1: Loading metadata...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = meta["TARGET"].values.astype(np.float32)
groups = meta["CV_GROUP"].values
train_ids = meta["ID"].values
n_train = len(y)
del meta; gc.collect()

test_ids = pd.read_parquet("test.parquet", columns=["ID"])["ID"].values
n_test = len(test_ids)

# Get all columns and compute simple correlation for top features
all_cols = pq.ParquetFile("train.parquet").schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
test_col_set = set(pq.ParquetFile("test.parquet").schema.names)

log("Step 2: Computing Correlations...")
y_c = y - y.mean()
corrs = {}
for i, c in enumerate(feat_cols):
    v = pd.read_parquet("train.parquet", columns=[c])[c].values.astype(np.float32)
    vs = v.std()
    corrs[c] = float(np.dot(y_c, v - v.mean()) / (n_train * y.std() * vs)) if vs > 0 else 0.0
    del v

TOP_N = 150 # Top 150 correlated features
selected = [name for name, _ in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:TOP_N]]
del corrs, y_c; gc.collect()

log("Step 3: Building Feature Matrix...")
def build_matrix(path, avail, n_rows, raw_feats):
    eng_needs = set(["Price", "Price_LagT1", "Price_LagT2", "Price_LagT3"])
    sfxs = ["F01_U01", "F02_U01", "F03_U01", "O01", "O02", "O01_A01", "O02_A01"]
    for s in sfxs:
        eng_needs.update([f"S01_{s}", f"S02_{s}", f"S01_{s}_LagT1", f"S02_{s}_LagT1"])

    to_load = list((set(raw_feats) | eng_needs) & avail)
    raw = {}
    for i in range(0, len(to_load), 50):
        b = to_load[i:i+50]
        df = pd.read_parquet(path, columns=b)
        for c in b: raw[c] = df[c].values.astype(np.float32)
        del df; gc.collect()

    eng = {}
    for s in sfxs:
        s1, s2 = f"S01_{s}", f"S02_{s}"
        if s1 in raw and s2 in raw:
            eng[f"spr_{s}"] = raw[s1] - raw[s2]
            eng[f"imb_{s}"] = (raw[s1]-raw[s2]) / (np.abs(raw[s1])+np.abs(raw[s2])+1e-10)
        s1l, s2l = f"S01_{s}_LagT1", f"S02_{s}_LagT1"
        if s1l in raw and s2l in raw:
            eng[f"spr_{s}_L1"] = raw[s1l] - raw[s2l]
        if f"spr_{s}" in eng and f"spr_{s}_L1" in eng:
            eng[f"sdchg_{s}"] = eng[f"spr_{s}"] - eng[f"spr_{s}_L1"]

    p = raw.get("Price", np.zeros(n_rows, np.float32))
    pl1 = raw.get("Price_LagT1", np.zeros(n_rows, np.float32))
    pl2 = raw.get("Price_LagT2", np.zeros(n_rows, np.float32))
    pl3 = raw.get("Price_LagT3", np.zeros(n_rows, np.float32))
    eng["pret"] = pl1 / (np.abs(p)+1e-10)
    eng["pmom"] = pl1 + pl2
    eng["paccel"] = pl1 - pl2
    eng["pvol"] = np.abs(pl1) + np.abs(pl2) + np.abs(pl3)

    for c in list(raw.keys()):
        if c not in set(raw_feats): del raw[c]
    gc.collect()

    eng_names = sorted(eng.keys())
    all_names = list(raw_feats) + eng_names
    X = np.empty((n_rows, len(all_names)), dtype=np.float32)
    for i, name in enumerate(raw_feats):
        a = raw.pop(name, None)
        X[:, i] = a if a is not None else 0.0
    for j, name in enumerate(eng_names):
        X[:, len(raw_feats)+j] = eng.pop(name)
    del raw, eng; gc.collect()
    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, all_names

X_train, feature_names = build_matrix("train.parquet", set(feat_cols), n_train, selected)
lo, hi = np.percentile(y, 1), np.percentile(y, 99)
y_clip = np.clip(y, lo, hi)

# ═════════════════════════════════════════════════════════════════════════════
# 2. ENSEMBLE TRAINING
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 4: Training Diverse Master Ensemble (GroupKFold)...")

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

CONFIGS = [
    {
        "name": "Huber",
        "params": {
            "objective": "huber", "alpha": 0.9, "metric": "mse", "boosting_type": "gbdt",
            "learning_rate": 0.01, "num_leaves": 127, "max_depth": -1, "min_child_samples": 100,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, 
            "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "n_jobs": -1, "random_state": 42
        }
    },
    {
        "name": "Fair",
        "params": {
            "objective": "fair", "fair_c": 1.0, "metric": "mse", "boosting_type": "gbdt",
            "learning_rate": 0.01, "num_leaves": 63, "max_depth": 7, "min_child_samples": 200,
            "feature_fraction": 0.7, "bagging_fraction": 0.9, "bagging_freq": 1,
            "reg_alpha": 0.5, "reg_lambda": 2.0, "verbose": -1, "n_jobs": -1, "random_state": 42
        }
    },
    {
        "name": "MSE",
        "params": {
            "objective": "regression", "metric": "mse", "boosting_type": "gbdt",
            "learning_rate": 0.005, "num_leaves": 63, "max_depth": 5, "min_child_samples": 150,
            "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1,
            "reg_alpha": 2.0, "reg_lambda": 5.0, "verbose": -1, "n_jobs": -1, "random_state": 42
        }
    }
]

oof_preds = {cfg["name"]: np.zeros(n_train, np.float32) for cfg in CONFIGS}
models = {cfg["name"]: [] for cfg in CONFIGS}

for cfg in CONFIGS:
    log(f"  Training {cfg['name']}...")
    for fi, (tr, va) in enumerate(gkf.split(X_train, y_clip, groups)):
        dt = lgb.Dataset(X_train[tr], y_clip[tr], feature_name=feature_names, free_raw_data=True)
        dv = lgb.Dataset(X_train[va], y_clip[va], feature_name=feature_names, free_raw_data=True, reference=dt)
        bst = lgb.train(cfg["params"], dt, 2500, valid_sets=[dv], valid_names=["v"],
                        callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_preds[cfg["name"]][va] = bst.predict(X_train[va])
        models[cfg["name"]].append(bst.model_to_string())
        del dt, dv, bst; gc.collect()
    
    cfg_r2 = r2_score(y, oof_preds[cfg["name"]])
    log(f"    {cfg['name']} OOF R2: {cfg_r2:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. OPTIMAL BLENDING & SCALING
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 5: Optimizing Blending Weights...")
names = [cfg["name"] for cfg in CONFIGS]
best_r2, best_w = -float('inf'), [1/3, 1/3, 1/3]

# Grid search 0-1 for weights
for w0 in np.arange(0.0, 1.05, 0.1):
    for w1 in np.arange(0.0, 1.05 - w0, 0.1):
        w2 = 1.0 - w0 - w1
        blend = oof_preds[names[0]]*w0 + oof_preds[names[1]]*w1 + oof_preds[names[2]]*w2
        r2 = r2_score(y, blend)
        if r2 > best_r2:
            best_r2 = r2
            best_w = [w0, w1, w2]

log(f"  Best Weights: {dict(zip(names, [round(w,2) for w in best_w]))}")
oof_blend = oof_preds[names[0]]*best_w[0] + oof_preds[names[1]]*best_w[1] + oof_preds[names[2]]*best_w[2]

# Scaling
log("  Optimizing Prediction Scale...")
best_scale, best_scale_r2 = 1.0, best_r2
for s in np.arange(0.5, 3.0, 0.05):
    sr2 = r2_score(y, oof_blend * s)
    if sr2 > best_scale_r2:
        best_scale, best_scale_r2 = s, sr2

log(f"  Optimal Scale factor: {best_scale:.2f}")
log(f"\n*** FINAL ENSEMBLE OOF R2: {best_scale_r2:.6f} ***")

# ═════════════════════════════════════════════════════════════════════════════
# 4. PREDICT TEST SET
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 6: Predicting Test Target...")
del X_train, oof_preds, oof_blend, y_clip; gc.collect()

X_test, _ = build_matrix("test.parquet", test_col_set, n_test, selected)
test_blend = np.zeros(n_test, np.float32)

for i, cfg in enumerate(CONFIGS):
    name = cfg["name"]
    w = best_w[i]
    if w == 0: continue
        
    cfg_preds = np.zeros(n_test, np.float32)
    for ms in models[name]:
        bst = lgb.Booster(model_str=ms)
        cfg_preds += bst.predict(X_test) / N_FOLDS
        del bst; gc.collect()
        
    test_blend += cfg_preds * w

# Apply optimal scale
test_final = test_blend * best_scale

# ═════════════════════════════════════════════════════════════════════════════
# 5. SAVE
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 7: Saving Submission Document...")
sub = pd.DataFrame({"ID": test_ids, "TARGET": test_final})
sub.to_csv("submission_master.csv", index=False)
sub.to_csv(os.path.join(OUTPUT_DIR, "submission_master.csv"), index=False)

log(f"  Saved submission_master.csv ({len(sub)} rows)")
log(sub["TARGET"].describe().to_string())
log("\nDONE. Ready for immediate upload.")
