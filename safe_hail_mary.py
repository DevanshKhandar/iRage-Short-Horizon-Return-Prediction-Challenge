"""
SAFE Final Hail Mary Pipeline
=============================
- 5-fold (to save time and memory)
- Test set prediction via row chunks to absolutely ensure no OOM.
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

TOP_N = 120 
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
        s1l, s2l = f"S01_{s}_LagT1", f"S02_{s}_LagT1"
        if s1l in raw and s2l in raw:
            eng[f"spr_{s}_L1"] = raw[s1l] - raw[s2l]
        if f"spr_{s}" in eng and f"spr_{s}_L1" in eng:
            eng[f"sdchg_{s}"] = eng[f"spr_{s}"] - eng[f"spr_{s}_L1"]

    p = raw.get("Price", np.zeros(n_rows, np.float32))
    pl1 = raw.get("Price_LagT1", np.zeros(n_rows, np.float32))
    pl2 = raw.get("Price_LagT2", np.zeros(n_rows, np.float32))
    eng["pret"] = pl1 / (np.abs(p)+1e-10)
    eng["pmom"] = pl1 + pl2

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

log("\nStep 4: Training 5-Fold Deep Ensemble...")

N_FOLDS = 5
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
        "name": "MSE",
        "params": {
            "objective": "regression", "metric": "root_mean_squared_error", "boosting_type": "gbdt",
            "learning_rate": 0.005, "num_leaves": 63, "max_depth": 7, "min_child_samples": 150,
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
        bst = lgb.train(cfg["params"], dt, 2000, valid_sets=[dv], valid_names=["v"],
                        callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_preds[cfg["name"]][va] = bst.predict(X_train[va])
        models[cfg["name"]].append(bst.model_to_string())
        del dt, dv, bst; gc.collect()
    
    cfg_r2 = r2_score(y, oof_preds[cfg["name"]])
    log(f"    {cfg['name']} OOF R2: {cfg_r2:.6f}")

log("\nStep 5: Blending...")
names = [cfg["name"] for cfg in CONFIGS]
best_w = [0.5, 0.5]
oof_blend = oof_preds[names[0]]*best_w[0] + oof_preds[names[1]]*best_w[1]

best_scale = 1.0; best_scale_r2 = r2_score(y, oof_blend)
for s in np.arange(0.5, 3.0, 0.1):
    sr2 = r2_score(y, oof_blend * s)
    if sr2 > best_scale_r2: best_scale, best_scale_r2 = s, sr2
         
log(f"  Final Ensemble OOF R2: {best_scale_r2:.6f} with Scale: {best_scale:.2f}")

log("\nStep 6: Predicting Test Target...")
del X_train, oof_preds, oof_blend, y_clip; gc.collect()

# We need to build matrix again, this time carefully
X_test, _ = build_matrix("test.parquet", test_col_set, n_test, selected)
test_blend = np.zeros(n_test, np.float32)

for i, cfg in enumerate(CONFIGS):
    name = cfg["name"]
    w = best_w[i]
    for ms in models[name]:
        bst = lgb.Booster(model_str=ms)
        # Predict in chunks to avoid memory errors!
        chunk_size = 50000
        for start_idx in range(0, n_test, chunk_size):
            end_idx = min(start_idx + chunk_size, n_test)
            test_blend[start_idx:end_idx] += bst.predict(X_test[start_idx:end_idx]) * w / N_FOLDS
        del bst; gc.collect()

test_final = test_blend * best_scale

log("\nStep 7: Applying Sequential Smoothing...")
test_meta["RAW_PRED"] = test_final
test_meta["original_idx"] = np.arange(len(test_meta))
test_meta = test_meta.sort_values(["CV_GROUP", "SO3_T"])

test_meta["pred_prev"] = test_meta.groupby("CV_GROUP")["RAW_PRED"].shift(1).fillna(test_meta["RAW_PRED"])
test_meta["pred_next"] = test_meta.groupby("CV_GROUP")["RAW_PRED"].shift(-1).fillna(test_meta["RAW_PRED"])

# Test 85% self, 15% momentum
test_meta["TARGET"] = 0.85 * test_meta["RAW_PRED"] + 0.075 * test_meta["pred_prev"] + 0.075 * test_meta["pred_next"]

test_sub = test_meta.sort_values("original_idx")[["ID", "TARGET"]]
test_sub.to_csv("submission_hail_mary.csv", index=False)
log(f"Saved submission_hail_mary.csv ({len(test_sub)} rows)")
log("DONE!")
