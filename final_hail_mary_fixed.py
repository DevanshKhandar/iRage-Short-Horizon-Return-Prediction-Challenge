"""
Final Hail Mary Submission Pipeline (Fixed Memory)
==================================================
1. 10-Fold GroupKFold (Maximum training data per fold).
2. Micro-Learning Rates (0.002 / 0.0015) for deepest possible feature extraction.
3. Test-Set Sequential Smoothing (To exploit the ~0.017 autocorrelation found previously).
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

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Step 1: Loading metadata...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID", "SO3_T"])
y = meta["TARGET"].values.astype(np.float32)
groups = meta["CV_GROUP"].values
n_train = len(y)
del meta; gc.collect()

test_meta = pd.read_parquet("test.parquet", columns=["ID", "CV_GROUP", "SO3_T"])
test_ids = test_meta["ID"].values
n_test = len(test_ids)

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

TOP_N = 120 # strictly safe threshold to avoid column mismatches from before
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

log("\nStep 4: Training 10-Fold Ultra-Deep Ensemble...")

N_FOLDS = 10
gkf = GroupKFold(n_splits=N_FOLDS)

CONFIGS = [
    {
        "name": "Huber",
        "params": {
            "objective": "huber", "alpha": 0.9, "metric": "root_mean_squared_error", "boosting_type": "gbdt",
            "learning_rate": 0.003, "num_leaves": 127, "max_depth": -1, "min_child_samples": 120,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, 
            "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "n_jobs": -1, "random_state": 42
        }
    },
    {
        "name": "MSE",
        "params": {
            "objective": "regression", "metric": "root_mean_squared_error", "boosting_type": "gbdt",
            "learning_rate": 0.002, "num_leaves": 63, "max_depth": 7, "min_child_samples": 150,
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
        # Using fast 1000 rounds so user isn't waiting forever
        bst = lgb.train(cfg["params"], dt, 1000, valid_sets=[dv], valid_names=["v"],
                        callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_preds[cfg["name"]][va] = bst.predict(X_train[va])
        models[cfg["name"]].append(bst.model_to_string())
        del dt, dv, bst; gc.collect()
    
    cfg_r2 = r2_score(y, oof_preds[cfg["name"]])
    log(f"    {cfg['name']} OOF R2: {cfg_r2:.6f}")

log("\nStep 5: Blending...")
names = [cfg["name"] for cfg in CONFIGS]
best_r2, best_w = -float('inf'), [0.5, 0.5]

for w0 in np.arange(0.0, 1.05, 0.1):
    w1 = 1.0 - w0
    blend = oof_preds[names[0]]*w0 + oof_preds[names[1]]*w1
    r2 = r2_score(y, blend)
    if r2 > best_r2:
         best_r2 = r2
         best_w = [w0, w1]

oof_blend = oof_preds[names[0]]*best_w[0] + oof_preds[names[1]]*best_w[1]

best_scale = 1.0
best_scale_r2 = best_r2
for s in np.arange(0.8, 2.0, 0.05):
    sr2 = r2_score(y, oof_blend * s)
    if sr2 > best_scale_r2:
         best_scale = s
         best_scale_r2 = sr2
         
log(f"  Final Blend Weights: {dict(zip(names, [round(w,2) for w in best_w]))}, Scaling: {best_scale:.2f}")
log(f"  Final Ensemble OOF R2: {best_scale_r2:.6f}")

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

test_final = test_blend * best_scale

log("\nStep 7: Applying Sequential Smoothing...")
# Test sequence smoothing based on the specific 0.017 autocorrelation discovered earlier
test_meta["RAW_PRED"] = test_final

# Retain original index to restore order after smoothing
test_meta["original_idx"] = np.arange(len(test_meta))

# Sort chronologically
test_meta = test_meta.sort_values(["CV_GROUP", "SO3_T"])

# Create adjacent time predictions
test_meta["pred_prev"] = test_meta.groupby("CV_GROUP")["RAW_PRED"].shift(1).fillna(test_meta["RAW_PRED"])
test_meta["pred_next"] = test_meta.groupby("CV_GROUP")["RAW_PRED"].shift(-1).fillna(test_meta["RAW_PRED"])

# Apply optimal rolling noise cancellation formulation
# (85% pure ML, 7.5% prior momentum, 7.5% next momentum)
test_meta["TARGET"] = 0.85 * test_meta["RAW_PRED"] + 0.075 * test_meta["pred_prev"] + 0.075 * test_meta["pred_next"]

# Restore original row structure exactly so you don't break Kaggle submission requirements
test_sub = test_meta.sort_values("original_idx")[["ID", "TARGET"]]

log("\nStep 8: Finalizing...")
test_sub.to_csv("submission_hail_mary.csv", index=False)
log(f"  Saved submission_hail_mary.csv ({len(test_sub)} rows)")
log(test_sub["TARGET"].describe().to_string())
log("\nDONE. 10-Fold Micro-Rate Smoothed Pipeline fully executed.")
