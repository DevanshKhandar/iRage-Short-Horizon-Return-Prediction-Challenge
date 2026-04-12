"""
Short-Horizon Return Prediction — Ultra Memory-Efficient Pipeline
==================================================================
For systems with ~4GB available RAM.
Strategy: Single LightGBM config, 1 seed, 5-fold GroupKFold.
Minimal footprint: train & predict fold-by-fold, never keep more than needed.
"""

import os
import gc
import time
import warnings
import json
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
# 1. LOAD METADATA & CORRELATIONS
# ═════════════════════════════════════════════════════════════════════════════
log("Step 1: Loading metadata & computing correlations...")
train_meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = train_meta["TARGET"].values.astype(np.float64)
groups = train_meta["CV_GROUP"].values
n_train = len(y)
del train_meta; gc.collect()

test_ids = pd.read_parquet("test.parquet", columns=["ID"])["ID"].values
n_test = len(test_ids)

all_cols = pq.ParquetFile("train.parquet").schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
test_col_set = set(pq.ParquetFile("test.parquet").schema.names)
log(f"  Train: {n_train}, Test: {n_test}, Features: {len(feat_cols)}")

# Correlations
log("  Computing correlations...")
y_c = (y - y.mean()).astype(np.float32)
y_s = float(y.std())
corrs = {}
for i, c in enumerate(feat_cols):
    v = pd.read_parquet("train.parquet", columns=[c])[c].values.astype(np.float32)
    vs = v.std()
    corrs[c] = float(np.dot(y_c, v - v.mean()) / (n_train * y_s * vs)) if vs > 0 else 0.0
    del v
    if (i + 1) % 100 == 0:
        log(f"    {i+1}/{len(feat_cols)}")
        gc.collect()

corrs_sorted = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
log(f"  Top: {corrs_sorted[0][0]} ({corrs_sorted[0][1]:.6f})")

TOP_N = 120
selected_raw = [name for name, _ in corrs_sorted[:TOP_N]]
log(f"  Selected top {TOP_N} features")
del corrs, corrs_sorted, y_c; gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# 2. BUILD FEATURE MATRIX FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def build_matrix(path, available_cols_set, n_rows, raw_features):
    """Build feature matrix: raw + engineered, memory-efficient."""
    # Determine extra columns needed for engineering
    eng_needs = set(["Price", "Price_LagT1", "Price_LagT2", "Price_LagT3"])
    suffixes = ["F01_U01", "F02_U01", "F03_U01", "O01", "O02", "O01_A01", "O02_A01"]
    for sfx in suffixes:
        eng_needs.add(f"S01_{sfx}"); eng_needs.add(f"S02_{sfx}")
        eng_needs.add(f"S01_{sfx}_LagT1"); eng_needs.add(f"S02_{sfx}_LagT1")
    for i in range(11):
        for pf in ["S03_D02_A09_A02", "S03_D02_V01_A01"]:
            eng_needs.add(f"{pf}_B{i:02d}_E{i:02d}_E{i+1:02d}")
            eng_needs.add(f"{pf}_B{i:02d}_E{i:02d}_E{i+1:02d}_LagT1")

    all_to_load = list((set(raw_features) | eng_needs) & available_cols_set)
    raw = {}
    for i in range(0, len(all_to_load), 50):
        batch = all_to_load[i:i+50]
        df = pd.read_parquet(path, columns=batch)
        for c in batch:
            raw[c] = df[c].values.astype(np.float32)
        del df; gc.collect()
    
    log(f"    Loaded {len(raw)} columns")
    
    # Engineer features
    eng = {}
    for sfx in suffixes:
        s1, s2 = f"S01_{sfx}", f"S02_{sfx}"
        if s1 in raw and s2 in raw:
            eng[f"spr_{sfx}"] = raw[s1] - raw[s2]
            eng[f"imb_{sfx}"] = (raw[s1] - raw[s2]) / (np.abs(raw[s1]) + np.abs(raw[s2]) + 1e-10)
        s1l, s2l = f"S01_{sfx}_LagT1", f"S02_{sfx}_LagT1"
        if s1l in raw and s2l in raw:
            eng[f"spr_{sfx}_L1"] = raw[s1l] - raw[s2l]

    # Spread changes
    for sfx in suffixes:
        c, l = f"spr_{sfx}", f"spr_{sfx}_L1"
        if c in eng and l in eng:
            eng[f"sdchg_{sfx}"] = eng[c] - eng[l]

    # Book depth
    a09_l, v01_l = [], []
    for i in range(11):
        a = f"S03_D02_A09_A02_B{i:02d}_E{i:02d}_E{i+1:02d}"
        v = f"S03_D02_V01_A01_B{i:02d}_E{i:02d}_E{i+1:02d}"
        if a in raw: a09_l.append(raw[a])
        if v in raw: v01_l.append(raw[v])
    if a09_l and v01_l:
        at, vt = sum(a09_l), sum(v01_l)
        eng["book_imb"] = (vt - at) / (vt + at + 1e-10)
        eng["tob_imb"] = (v01_l[0] - a09_l[0]) / (np.abs(v01_l[0]) + np.abs(a09_l[0]) + 1e-10)

    # OFI
    a09_lag, v01_lag = [], []
    for i in range(11):
        a = f"S03_D02_A09_A02_B{i:02d}_E{i:02d}_E{i+1:02d}_LagT1"
        v = f"S03_D02_V01_A01_B{i:02d}_E{i:02d}_E{i+1:02d}_LagT1"
        if a in raw: a09_lag.append(raw[a])
        if v in raw: v01_lag.append(raw[v])
    if a09_lag and v01_lag:
        eng["ofi_L1"] = sum(v01_lag) - sum(a09_lag)

    # Price features
    p = raw.get("Price", np.zeros(n_rows, np.float32))
    pl1 = raw.get("Price_LagT1", np.zeros(n_rows, np.float32))
    pl2 = raw.get("Price_LagT2", np.zeros(n_rows, np.float32))
    pl3 = raw.get("Price_LagT3", np.zeros(n_rows, np.float32))
    eng["pret"] = pl1 / (np.abs(p) + 1e-10)
    eng["pmom"] = pl1 + pl2
    eng["paccel"] = pl1 - pl2
    eng["pvol"] = np.abs(pl1) + np.abs(pl2) + np.abs(pl3)
    if "book_imb" in eng:
        eng["p_x_bimb"] = p * eng["book_imb"]
    if "ofi_L1" in eng:
        eng["ofi_x_p"] = eng["ofi_L1"] * p

    # Free extra columns
    for c in list(raw.keys()):
        if c not in set(raw_features):
            del raw[c]
    gc.collect()

    # Build matrix
    eng_names = sorted(eng.keys())
    all_names = list(raw_features) + eng_names
    n_feat = len(all_names)
    log(f"    Engineered {len(eng_names)} features, total: {n_feat}")

    X = np.empty((n_rows, n_feat), dtype=np.float32)
    for i, name in enumerate(raw_features):
        arr = raw.pop(name, None)
        X[:, i] = arr if arr is not None else 0.0
        if arr is not None: del arr
    del raw; gc.collect()

    for j, name in enumerate(eng_names):
        X[:, len(raw_features) + j] = eng.pop(name)
    del eng; gc.collect()

    mask = ~np.isfinite(X)
    if mask.any(): X[mask] = 0.0
    del mask; gc.collect()

    return X, all_names

# ═════════════════════════════════════════════════════════════════════════════
# 3. BUILD TRAIN MATRIX
# ═════════════════════════════════════════════════════════════════════════════
log("Step 2: Building train matrix...")
X_train, feature_names = build_matrix("train.parquet", set(feat_cols), n_train, selected_raw)
log(f"  X_train: {X_train.shape} ({X_train.nbytes / 1e9:.2f} GB)")

# Target preprocessing
lo, hi = np.percentile(y, 1), np.percentile(y, 99)
y_clip = np.clip(y, lo, hi)
log(f"  Target clipped [{lo:.4f}, {hi:.4f}]")

# ═════════════════════════════════════════════════════════════════════════════
# 4. TRAIN & PREDICT FOLD-BY-FOLD
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 3: Training fold-by-fold...")

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

# Single strong configuration — less complex = less memory
params = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 100,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_gain_to_split": 0.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

oof_preds = np.zeros(n_train, np.float64)
fold_models = []  # store model strings for test prediction

for fi, (tr, va) in enumerate(gkf.split(X_train, y_clip, groups)):
    log(f"  Fold {fi+1}/{N_FOLDS} (train={len(tr)}, val={len(va)})")
    
    dt = lgb.Dataset(X_train[tr], y_clip[tr], feature_name=feature_names, free_raw_data=True)
    dv = lgb.Dataset(X_train[va], y_clip[va], feature_name=feature_names, free_raw_data=True, reference=dt)
    
    bst = lgb.train(
        params, dt, 5000,
        valid_sets=[dv], valid_names=["v"],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)]
    )
    
    oof_preds[va] = bst.predict(X_train[va])
    fold_r2 = r2_score(y[va], oof_preds[va])
    log(f"    R2={fold_r2:.6f}, best_iter={bst.best_iteration}")
    
    # Save model as string (compact)
    fold_models.append(bst.model_to_string())
    del dt, dv, bst; gc.collect()

oof_r2 = r2_score(y, oof_preds)
log(f"\n  Overall OOF R2: {oof_r2:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. ALSO TRAIN HUBER (second config for diversity)
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 4: Training Huber model...")

params_huber = {
    "objective": "huber",
    "alpha": 0.9,
    "metric": "mse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_gain_to_split": 0.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

oof_huber = np.zeros(n_train, np.float64)
huber_models = []

for fi, (tr, va) in enumerate(gkf.split(X_train, y_clip, groups)):
    log(f"  Fold {fi+1}/{N_FOLDS}")
    
    dt = lgb.Dataset(X_train[tr], y_clip[tr], feature_name=feature_names, free_raw_data=True)
    dv = lgb.Dataset(X_train[va], y_clip[va], feature_name=feature_names, free_raw_data=True, reference=dt)
    
    bst = lgb.train(
        params_huber, dt, 5000,
        valid_sets=[dv], valid_names=["v"],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)]
    )
    
    oof_huber[va] = bst.predict(X_train[va])
    fold_r2 = r2_score(y[va], oof_huber[va])
    log(f"    R2={fold_r2:.6f}, best_iter={bst.best_iteration}")
    
    huber_models.append(bst.model_to_string())
    del dt, dv, bst; gc.collect()

huber_r2 = r2_score(y, oof_huber)
log(f"  Huber OOF R2: {huber_r2:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. BLEND OOF & DETERMINE BEST WEIGHTS
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 5: Finding optimal blend...")

log(f"  MSE model R2:   {oof_r2:.6f}")
log(f"  Huber model R2: {huber_r2:.6f}")

best_r2 = max(oof_r2, huber_r2)
best_w = 1.0 if oof_r2 >= huber_r2 else 0.0

for w in np.arange(0, 1.01, 0.05):
    blend = oof_preds * w + oof_huber * (1 - w)
    r2 = r2_score(y, blend)
    if r2 > best_r2:
        best_r2 = r2
        best_w = w

log(f"  Best weight (MSE): {best_w:.2f}, Blend R2: {best_r2:.6f}")

oof_final = oof_preds * best_w + oof_huber * (1 - best_w)
final_r2 = r2_score(y, oof_final)

# Safety: check vs zero prediction
zero_r2 = r2_score(y, np.zeros(n_train))
log(f"  Zero-pred R2: {zero_r2:.6f}")

scale_factor = 1.0
if final_r2 < zero_r2:
    log("  WARNING: worse than zero, finding shrinkage")
    best_a, best_ar2 = 0.0, zero_r2
    for a in np.arange(0, 1.01, 0.01):
        ar2 = r2_score(y, oof_final * a)
        if ar2 > best_ar2:
            best_ar2, best_a = ar2, a
    scale_factor = best_a
    final_r2 = best_ar2
    log(f"  Shrinkage: {best_a:.2f}, R2: {best_ar2:.6f}")
else:
    best_s, best_sr2 = 1.0, final_r2
    for s in np.arange(0.8, 1.21, 0.01):
        sr2 = r2_score(y, oof_final * s)
        if sr2 > best_sr2:
            best_sr2, best_s = sr2, s
    if best_s != 1.0:
        scale_factor = best_s
        final_r2 = best_sr2
        log(f"  Scale {best_s:.2f}: R2 -> {best_sr2:.6f}")

log(f"\n  *** FINAL OOF R2: {final_r2:.6f} ***")

# Per-fold
log("  Per-fold R2 (blended & scaled):")
for fi, (tr, va) in enumerate(gkf.split(X_train, y, groups)):
    fr2 = r2_score(y[va], oof_final[va] * scale_factor)
    log(f"    Fold {fi+1}: R2={fr2:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. FREE TRAIN, BUILD TEST, PREDICT
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 6: Predicting test set...")

del X_train, oof_preds, oof_huber, oof_final, y_clip
gc.collect()

X_test, _ = build_matrix("test.parquet", test_col_set, n_test, selected_raw)
log(f"  X_test: {X_test.shape} ({X_test.nbytes / 1e9:.2f} GB)")

# Predict with MSE models
test_mse = np.zeros(n_test, np.float64)
for model_str in fold_models:
    bst = lgb.Booster(model_str=model_str)
    test_mse += bst.predict(X_test) / N_FOLDS
    del bst; gc.collect()
del fold_models; gc.collect()

# Predict with Huber models
test_huber = np.zeros(n_test, np.float64)
for model_str in huber_models:
    bst = lgb.Booster(model_str=model_str)
    test_huber += bst.predict(X_test) / N_FOLDS
    del bst; gc.collect()
del huber_models; gc.collect()

# Blend & scale
test_final = (test_mse * best_w + test_huber * (1 - best_w)) * scale_factor

log(f"  Test pred: mean={test_final.mean():.6f}, std={test_final.std():.6f}")
log(f"  Test range: [{test_final.min():.6f}, {test_final.max():.6f}]")

# ═════════════════════════════════════════════════════════════════════════════
# 8. SAVE
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 7: Saving submission...")

sub = pd.DataFrame({"ID": test_ids, "TARGET": test_final})
sub.to_csv("submission.csv", index=False)
sub.to_csv(os.path.join(OUTPUT_DIR, "submission_v3.csv"), index=False)

log(f"  Saved submission.csv ({len(sub)} rows)")
log(f"  Stats:")
log(sub["TARGET"].describe().to_string())

cv_info = {
    "final_r2": round(float(final_r2), 6),
    "mse_r2": round(float(oof_r2), 6),
    "huber_r2": round(float(huber_r2), 6),
    "blend_weight_mse": round(float(best_w), 3),
    "scale_factor": round(float(scale_factor), 3),
    "n_features": len(feature_names),
}
with open(os.path.join(OUTPUT_DIR, "cv_results_v3.json"), "w") as f:
    json.dump(cv_info, f, indent=2)

log("\n" + "=" * 60)
log("PIPELINE COMPLETE!")
log(f"Final OOF R2: {final_r2:.6f}")
log("=" * 60)
