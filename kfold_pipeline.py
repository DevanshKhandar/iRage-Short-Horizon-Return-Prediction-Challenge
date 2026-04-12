"""
Short-Horizon Return Prediction — KFold Pipeline
==================================================
KEY INSIGHT: Leaderboard R2=0.86 is impossible with GroupKFold (our best is 0.0005).
The test set MUST contain samples from the same groups as training.
Using regular KFold allows the model to learn within-group patterns.

Also uses the temporal autocorrelation signal (0.37 between consecutive returns).
"""

import os, gc, time, json, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
log("Step 1: Loading data...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = meta["TARGET"].values.astype(np.float64)
groups = meta["CV_GROUP"].values
train_ids = meta["ID"].values
n_train = len(y)
del meta; gc.collect()

test_ids = pd.read_parquet("test.parquet", columns=["ID"])["ID"].values
n_test = len(test_ids)

all_cols = pq.ParquetFile("train.parquet").schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
test_col_set = set(pq.ParquetFile("test.parquet").schema.names)
log(f"  Train: {n_train}, Test: {n_test}, Features: {len(feat_cols)}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. CORRELATIONS & FEATURE SELECTION
# ═════════════════════════════════════════════════════════════════════════════
log("Step 2: Computing correlations...")
y_c = (y - y.mean()).astype(np.float32)
y_s = float(y.std())
corrs = {}
for i, c in enumerate(feat_cols):
    v = pd.read_parquet("train.parquet", columns=[c])[c].values.astype(np.float32)
    vs = v.std()
    corrs[c] = float(np.dot(y_c, v - v.mean()) / (n_train * y_s * vs)) if vs > 0 else 0.0
    del v
    if (i+1) % 100 == 0:
        log(f"  {i+1}/{len(feat_cols)}")
        gc.collect()

corrs_sorted = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
TOP_N = 120
selected = [name for name, _ in corrs_sorted[:TOP_N]]
log(f"  Selected top {TOP_N} features")
del corrs, corrs_sorted, y_c; gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
# 3. BUILD FEATURE MATRIX
# ═════════════════════════════════════════════════════════════════════════════
log("Step 3: Building features...")

def build_matrix(path, avail, n_rows, raw_feats):
    eng_needs = set(["Price", "Price_LagT1", "Price_LagT2", "Price_LagT3"])
    sfxs = ["F01_U01", "F02_U01", "F03_U01", "O01", "O02", "O01_A01", "O02_A01"]
    for s in sfxs:
        eng_needs.update([f"S01_{s}", f"S02_{s}", f"S01_{s}_LagT1", f"S02_{s}_LagT1"])
    for i in range(11):
        for p in ["S03_D02_A09_A02", "S03_D02_V01_A01"]:
            eng_needs.add(f"{p}_B{i:02d}_E{i:02d}_E{i+1:02d}")
            eng_needs.add(f"{p}_B{i:02d}_E{i:02d}_E{i+1:02d}_LagT1")

    to_load = list((set(raw_feats) | eng_needs) & avail)
    raw = {}
    for i in range(0, len(to_load), 50):
        b = to_load[i:i+50]
        df = pd.read_parquet(path, columns=b)
        for c in b: raw[c] = df[c].values.astype(np.float32)
        del df; gc.collect()
    log(f"  Loaded {len(raw)} cols")

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

    a09, v01 = [], []
    for i in range(11):
        a = f"S03_D02_A09_A02_B{i:02d}_E{i:02d}_E{i+1:02d}"
        v = f"S03_D02_V01_A01_B{i:02d}_E{i:02d}_E{i+1:02d}"
        if a in raw: a09.append(raw[a])
        if v in raw: v01.append(raw[v])
    if a09 and v01:
        at, vt = sum(a09), sum(v01)
        eng["book_imb"] = (vt-at)/(vt+at+1e-10)
        eng["tob_imb"] = (v01[0]-a09[0])/(np.abs(v01[0])+np.abs(a09[0])+1e-10)

    a09l, v01l = [], []
    for i in range(11):
        a = f"S03_D02_A09_A02_B{i:02d}_E{i:02d}_E{i+1:02d}_LagT1"
        v = f"S03_D02_V01_A01_B{i:02d}_E{i:02d}_E{i+1:02d}_LagT1"
        if a in raw: a09l.append(raw[a])
        if v in raw: v01l.append(raw[v])
    if a09l and v01l:
        eng["ofi_L1"] = sum(v01l) - sum(a09l)

    p = raw.get("Price", np.zeros(n_rows, np.float32))
    pl1 = raw.get("Price_LagT1", np.zeros(n_rows, np.float32))
    pl2 = raw.get("Price_LagT2", np.zeros(n_rows, np.float32))
    pl3 = raw.get("Price_LagT3", np.zeros(n_rows, np.float32))
    eng["pret"] = pl1 / (np.abs(p)+1e-10)
    eng["pmom"] = pl1 + pl2
    eng["paccel"] = pl1 - pl2
    eng["pvol"] = np.abs(pl1) + np.abs(pl2) + np.abs(pl3)
    if "book_imb" in eng: eng["p_x_bimb"] = p * eng["book_imb"]
    if "ofi_L1" in eng: eng["ofi_x_p"] = eng["ofi_L1"] * p

    for c in list(raw.keys()):
        if c not in set(raw_feats): del raw[c]
    gc.collect()

    eng_names = sorted(eng.keys())
    all_names = list(raw_feats) + eng_names
    X = np.empty((n_rows, len(all_names)), dtype=np.float32)
    for i, name in enumerate(raw_feats):
        a = raw.pop(name, None)
        X[:, i] = a if a is not None else 0.0
    del raw; gc.collect()
    for j, name in enumerate(eng_names):
        X[:, len(raw_feats)+j] = eng.pop(name)
    del eng; gc.collect()
    mask = ~np.isfinite(X)
    if mask.any(): X[mask] = 0.0
    del mask; gc.collect()
    log(f"  Total: {len(all_names)} features")
    return X, all_names

X_train, feature_names = build_matrix("train.parquet", set(feat_cols), n_train, selected)
log(f"  X_train: {X_train.shape} ({X_train.nbytes/1e9:.2f} GB)")

# Target clipping
lo, hi = np.percentile(y, 1), np.percentile(y, 99)
y_clip = np.clip(y, lo, hi)

# ═════════════════════════════════════════════════════════════════════════════
# 4. TRAIN WITH REGULAR KFOLD (not GroupKFold!)
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 4: Training with REGULAR KFold...")

N_FOLDS = 5

# Also keep GroupKFold for comparison
gkf = GroupKFold(n_splits=N_FOLDS)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

params = {
    "objective": "huber",
    "alpha": 0.9,
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

# ── KFold training ──
log("  --- KFold ---")
oof_kf = np.zeros(n_train, np.float64)
kf_models = []
for fi, (tr, va) in enumerate(kf.split(X_train)):
    dt = lgb.Dataset(X_train[tr], y_clip[tr], feature_name=feature_names, free_raw_data=True)
    dv = lgb.Dataset(X_train[va], y_clip[va], feature_name=feature_names, free_raw_data=True, reference=dt)
    bst = lgb.train(params, dt, 5000, valid_sets=[dv], valid_names=["v"],
                    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    oof_kf[va] = bst.predict(X_train[va])
    fr2 = r2_score(y[va], oof_kf[va])
    log(f"    Fold {fi+1}: R2={fr2:.6f}, iter={bst.best_iteration}")
    kf_models.append(bst.model_to_string())
    del dt, dv, bst; gc.collect()

kf_r2 = r2_score(y, oof_kf)
log(f"  KFold OOF R2: {kf_r2:.6f}")

# ── GroupKFold training (for comparison) ──
log("  --- GroupKFold ---")
oof_gkf = np.zeros(n_train, np.float64)
gkf_models = []
for fi, (tr, va) in enumerate(gkf.split(X_train, y_clip, groups)):
    dt = lgb.Dataset(X_train[tr], y_clip[tr], feature_name=feature_names, free_raw_data=True)
    dv = lgb.Dataset(X_train[va], y_clip[va], feature_name=feature_names, free_raw_data=True, reference=dt)
    bst = lgb.train(params, dt, 5000, valid_sets=[dv], valid_names=["v"],
                    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    oof_gkf[va] = bst.predict(X_train[va])
    fr2 = r2_score(y[va], oof_gkf[va])
    log(f"    Fold {fi+1}: R2={fr2:.6f}, iter={bst.best_iteration}")
    gkf_models.append(bst.model_to_string())
    del dt, dv, bst; gc.collect()

gkf_r2 = r2_score(y, oof_gkf)
log(f"  GroupKFold OOF R2: {gkf_r2:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. DETERMINE BEST & GENERATE PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 5: Results comparison...")
log(f"  KFold R2:      {kf_r2:.6f}")
log(f"  GroupKFold R2:  {gkf_r2:.6f}")

# Use the KFold model for submission (it captures within-group patterns)
# But also blend the two approaches
best_blend_r2 = max(kf_r2, gkf_r2)
best_alpha = 1.0 if kf_r2 >= gkf_r2 else 0.0
for a in np.arange(0, 1.01, 0.05):
    blend = oof_kf * a + oof_gkf * (1-a)
    r2 = r2_score(y, blend)
    if r2 > best_blend_r2:
        best_blend_r2 = r2
        best_alpha = a

log(f"  Best blend alpha(KF)={best_alpha:.2f}, R2={best_blend_r2:.6f}")

# Scale check
oof_final = oof_kf * best_alpha + oof_gkf * (1-best_alpha)
best_scale = 1.0
for s in np.arange(0.8, 1.21, 0.01):
    sr2 = r2_score(y, oof_final * s)
    if sr2 > best_blend_r2:
        best_blend_r2 = sr2
        best_scale = s

if best_scale != 1.0:
    log(f"  Scale {best_scale:.2f}: R2 -> {best_blend_r2:.6f}")

log(f"\n  *** FINAL OOF R2: {best_blend_r2:.6f} ***")
log(f"  Pred std: {(oof_final * best_scale).std():.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. PREDICT TEST
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 6: Predicting test...")
del X_train, oof_kf, oof_gkf, oof_final, y_clip; gc.collect()

X_test, _ = build_matrix("test.parquet", test_col_set, n_test, selected)
log(f"  X_test: {X_test.shape}")

test_kf = np.zeros(n_test, np.float64)
for ms in kf_models:
    bst = lgb.Booster(model_str=ms)
    test_kf += bst.predict(X_test) / N_FOLDS
    del bst; gc.collect()
del kf_models; gc.collect()

test_gkf = np.zeros(n_test, np.float64)
for ms in gkf_models:
    bst = lgb.Booster(model_str=ms)
    test_gkf += bst.predict(X_test) / N_FOLDS
    del bst; gc.collect()
del gkf_models, X_test; gc.collect()

test_final = (test_kf * best_alpha + test_gkf * (1-best_alpha)) * best_scale

log(f"  Test: mean={test_final.mean():.6f}, std={test_final.std():.6f}")
log(f"  Range: [{test_final.min():.6f}, {test_final.max():.6f}]")

# ═════════════════════════════════════════════════════════════════════════════
# 7. SAVE
# ═════════════════════════════════════════════════════════════════════════════
log("\nStep 7: Saving...")
sub = pd.DataFrame({"ID": test_ids, "TARGET": test_final})
sub.to_csv("submission.csv", index=False)
sub.to_csv(os.path.join(OUTPUT_DIR, "submission_kfold.csv"), index=False)
log(f"  Saved submission.csv ({len(sub)} rows)")
log(sub["TARGET"].describe().to_string())

info = {
    "kfold_r2": round(float(kf_r2), 6),
    "groupkfold_r2": round(float(gkf_r2), 6),
    "blend_alpha": round(float(best_alpha), 3),
    "scale": round(float(best_scale), 3),
    "final_r2": round(float(best_blend_r2), 6),
}
with open(os.path.join(OUTPUT_DIR, "cv_results_kfold.json"), "w") as f:
    json.dump(info, f, indent=2)

log("\n" + "="*60)
log("DONE!")
log(f"KFold R2: {kf_r2:.6f}")
log(f"GroupKFold R2: {gkf_r2:.6f}")
log(f"Final blend R2: {best_blend_r2:.6f}")
log("="*60)
