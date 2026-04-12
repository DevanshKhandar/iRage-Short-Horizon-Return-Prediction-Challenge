"""
Fast KFold Pipeline
====================
User requested to see the CSV now.
Using regular KFold (R2 ~ 0.17) and fast learning rate.
"""

import os, gc, time, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Step 1: Loading data...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "ID"])
y = meta["TARGET"].values.astype(np.float32)
train_ids = meta["ID"].values
n_train = len(y)
del meta; gc.collect()

test_ids = pd.read_parquet("test.parquet", columns=["ID"])["ID"].values
n_test = len(test_ids)

all_cols = pq.ParquetFile("train.parquet").schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]
test_col_set = set(pq.ParquetFile("test.parquet").schema.names)

log("Step 2: Computing Correlations (fast proxy)...")
y_c = y - y.mean()
# only need top ~80 features for speed
corrs = {}
for i, c in enumerate(feat_cols):
    v = pd.read_parquet("train.parquet", columns=[c])[c].values.astype(np.float32)
    vs = v.std()
    corrs[c] = float(np.dot(y_c, v - v.mean()) / (n_train * y.std() * vs)) if vs > 0 else 0.0
    del v

corrs_sorted = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
TOP_N = 80
selected = [name for name, _ in corrs_sorted[:TOP_N]]
del corrs, corrs_sorted, y_c; gc.collect()

log("Step 3: Building Features...")
def build_matrix(path, avail, n_rows, raw_feats):
    eng_needs = set(["Price", "Price_LagT1", "Price_LagT2", "Price_LagT3"])
    sfxs = ["F01_U01", "O01", "O01_A01"]
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

log("\nStep 4: Training Fast KFold...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

params = {
    "objective": "huber",
    "alpha": 0.9,
    "metric": "mse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

oof_kf = np.zeros(n_train, np.float32)
kf_models = []
for fi, (tr, va) in enumerate(kf.split(X_train)):
    dt = lgb.Dataset(X_train[tr], y_clip[tr], feature_name=feature_names, free_raw_data=True)
    dv = lgb.Dataset(X_train[va], y_clip[va], feature_name=feature_names, free_raw_data=True, reference=dt)
    bst = lgb.train(params, dt, 1000, valid_sets=[dv], valid_names=["v"],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_kf[va] = bst.predict(X_train[va])
    fr2 = r2_score(y[va], oof_kf[va])
    log(f"  Fold {fi+1}: R2={fr2:.6f}, iter={bst.best_iteration}")
    kf_models.append(bst.model_to_string())
    del dt, dv, bst; gc.collect()

kf_r2 = r2_score(y, oof_kf)
log(f"\nOverall KFold R2: {kf_r2:.6f}")

log("\nStep 5: Predict Test...")
del X_train, oof_kf, y_clip; gc.collect()

X_test, _ = build_matrix("test.parquet", test_col_set, n_test, selected)
test_preds = np.zeros(n_test, np.float32)
for ms in kf_models:
    bst = lgb.Booster(model_str=ms)
    test_preds += bst.predict(X_test) / 5
    del bst; gc.collect()

log("\nStep 6: Saving...")
sub = pd.DataFrame({"ID": test_ids, "TARGET": test_preds})
sub.to_csv("submission.csv", index=False)
sub.to_csv(os.path.join(OUTPUT_DIR, "submission_fast_kfold.csv"), index=False)

log(f"Saved submission.csv ({len(sub)} rows)")
log(sub["TARGET"].describe().to_string())
log("Done!")
