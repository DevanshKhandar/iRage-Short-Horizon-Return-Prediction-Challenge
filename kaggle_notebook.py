"""
Short-Horizon Return Prediction — Kaggle Submission Notebook
=============================================================
Competition-winning ensemble: 3 LightGBM configs x 3 seeds x 5 folds = 45 models
- Feature selection by target correlation (|corr| > 0.002)
- Target winsorization P1/P99
- Huber + MSE + Fair loss diversity
- Optimal weight blending with prediction scaling

Copy this entire file into a Kaggle notebook code cell.
"""

import os, gc, numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb

# ─── Config ──────────────────────────────────────────────────────────────────
INPUT = "/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage"
# INPUT = "."  # Uncomment for local run
SEED, N_FOLDS = 42, 5

CONFIGS = [
    {"name": "huber_shallow", "params": {"objective": "huber", "alpha": 0.5, "metric": "mse", "boosting_type": "gbdt", "learning_rate": 0.01, "num_leaves": 31, "max_depth": 5, "min_child_samples": 200, "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1, "reg_alpha": 1.0, "reg_lambda": 5.0, "min_gain_to_split": 0.01, "verbose": -1, "n_jobs": -1}, "rounds": 10000, "es": 300, "seeds": [42,123,456], "weight": 0.10},
    {"name": "mse_deep", "params": {"objective": "regression", "metric": "mse", "boosting_type": "gbdt", "learning_rate": 0.005, "num_leaves": 63, "max_depth": 7, "min_child_samples": 300, "feature_fraction": 0.4, "bagging_fraction": 0.6, "bagging_freq": 1, "reg_alpha": 2.0, "reg_lambda": 10.0, "min_gain_to_split": 0.005, "verbose": -1, "n_jobs": -1}, "rounds": 10000, "es": 300, "seeds": [42,789,101], "weight": 0.75},
    {"name": "fair_reg", "params": {"objective": "fair", "fair_c": 1.0, "metric": "mse", "boosting_type": "gbdt", "learning_rate": 0.01, "num_leaves": 15, "max_depth": 4, "min_child_samples": 500, "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, "reg_alpha": 5.0, "reg_lambda": 20.0, "min_gain_to_split": 0.02, "verbose": -1, "n_jobs": -1}, "rounds": 10000, "es": 300, "seeds": [42,202,303], "weight": 0.15},
]

# ─── 1. Load & compute correlations ──────────────────────────────────────────
print("Loading data...")
train_path = os.path.join(INPUT, "train.parquet")
test_path = os.path.join(INPUT, "test.parquet")

meta = pd.read_parquet(train_path, columns=["TARGET","CV_GROUP","ID"])
y = meta["TARGET"].values.astype(np.float32)
groups = meta["CV_GROUP"].values.copy()
n = len(y)
del meta; gc.collect()

all_cols = pq.ParquetFile(train_path).schema.names
feat_cols = [c for c in all_cols if c not in {"ID","CV_GROUP","TARGET"}]

print("Computing feature correlations...")
ym, ys, yc = y.mean(), y.std(), y - y.mean()
corrs = {}
for i, c in enumerate(feat_cols):
    v = pd.read_parquet(train_path, columns=[c])[c].values.astype(np.float32)
    vs = v.std()
    corrs[c] = float(np.dot(yc, v - v.mean()) / (n * ys * vs)) if vs > 0 else 0.0
    del v
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(feat_cols)}"); gc.collect()

selected = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
sel_feats = [name for name, corr in selected if abs(corr) > 0.002]
print(f"Selected {len(sel_feats)} features")

# ─── 2. Aggregate lag features ───────────────────────────────────────────────
all_l1 = [c for c in feat_cols if "_LagT1" in c]
all_l2 = [c for c in feat_cols if "_LagT2" in c]
all_l3 = [c for c in feat_cols if "_LagT3" in c]
extra = ["lag_t1_mean","lag_t1_std","lag_t2_mean","lag_t2_std","lag_t3_mean","lag_t3_std","lag_accel","lag_jerk","lag_ratio","lag_consistency"]

def agg(path, cols, l1, l2, l3):
    nn = pq.read_metadata(path).num_rows
    r = np.zeros((nn, 10), np.float32)
    for idx, ll in enumerate([l1, l2, l3]):
        s, sq = np.zeros(nn, np.float32), np.zeros(nn, np.float32)
        for c in ll:
            if c in cols:
                v = pd.read_parquet(path, columns=[c])[c].values.astype(np.float32)
                s += v; sq += v*v; del v
        m = s/max(len(ll),1); sd = np.sqrt(np.maximum(sq/max(len(ll),1)-m**2, 0))
        r[:, idx*2] = m; r[:, idx*2+1] = sd; del s, sq
    r[:,6] = r[:,0]-r[:,2]; r[:,7] = r[:,6]-(r[:,2]-r[:,4])
    r[:,8] = r[:,0]/(np.abs(r[:,2])+1e-10); r[:,9] = r[:,1]/(r[:,3]+1e-10)
    return r

# ─── 3. Build matrices ──────────────────────────────────────────────────────
print("Building train matrix...")
n_sel, n_ext, n_tot = len(sel_feats), len(extra), len(sel_feats)+len(extra)
train_agg = agg(train_path, feat_cols, all_l1, all_l2, all_l3)
X_train = np.empty((n, n_tot), np.float32)
for i, c in enumerate(sel_feats):
    X_train[:,i] = pd.read_parquet(train_path, columns=[c])[c].values.astype(np.float32)
    if (i+1)%100==0: gc.collect()
X_train[:, n_sel:] = train_agg; del train_agg; gc.collect()

print("Building test matrix...")
test_ids = pd.read_parquet(test_path, columns=["ID"])["ID"].values.copy()
n_test = len(test_ids)
test_cols = pq.ParquetFile(test_path).schema.names
test_agg = agg(test_path, test_cols, all_l1, all_l2, all_l3)
X_test = np.empty((n_test, n_tot), np.float32)
for i, c in enumerate(sel_feats):
    X_test[:,i] = pd.read_parquet(test_path, columns=[c])[c].values.astype(np.float32) if c in test_cols else 0.0
X_test[:, n_sel:] = test_agg; del test_agg; gc.collect()

fnames = sel_feats + extra
print(f"Features: {n_tot}")

# ─── 4. Target clipping ─────────────────────────────────────────────────────
lo, hi = np.percentile(y, 1), np.percentile(y, 99)
y_clip = np.clip(y, lo, hi)
print(f"Target clipped to [{lo:.4f}, {hi:.4f}]")

# ─── 5. Ensemble training ───────────────────────────────────────────────────
print("\nTraining ensemble...")
gkf = GroupKFold(n_splits=N_FOLDS)
all_oof, all_test = {}, {}

for cfg in CONFIGS:
    nm = cfg["name"]
    print(f"\n  Config: {nm}")
    c_oof = np.zeros(n, np.float64); c_test = np.zeros(n_test, np.float64)
    for si, seed in enumerate(cfg["seeds"]):
        p = {**cfg["params"], "random_state": seed}
        s_oof = np.zeros(n, np.float64); s_test = np.zeros(n_test, np.float64)
        for fi, (tr, va) in enumerate(gkf.split(X_train, y_clip, groups)):
            dt = lgb.Dataset(X_train[tr], y_clip[tr], feature_name=fnames, free_raw_data=True)
            dv = lgb.Dataset(X_train[va], y_clip[va], feature_name=fnames, free_raw_data=True, reference=dt)
            b = lgb.train(p, dt, cfg["rounds"], valid_sets=[dv], valid_names=["v"],
                          callbacks=[lgb.early_stopping(cfg["es"], verbose=False), lgb.log_evaluation(0)])
            s_oof[va] = b.predict(X_train[va])
            s_test += b.predict(X_test) / N_FOLDS
            del dt, dv, b; gc.collect()
        print(f"    Seed {seed}: R2={r2_score(y, s_oof):.6f}")
        c_oof += s_oof/len(cfg["seeds"]); c_test += s_test/len(cfg["seeds"])
    print(f"    Config R2: {r2_score(y, c_oof):.6f}")
    all_oof[nm] = c_oof; all_test[nm] = c_test

# ─── 6. Optimal blending ────────────────────────────────────────────────────
print("\nOptimizing weights...")
names = [c["name"] for c in CONFIGS]
best_r2, best_w = -999, [c["weight"] for c in CONFIGS]
for w0 in np.arange(0.05, 0.95, 0.05):
    for w1 in np.arange(0.05, 0.95-w0, 0.05):
        w2 = 1-w0-w1
        if w2 < 0.05: continue
        trial = all_oof[names[0]]*w0 + all_oof[names[1]]*w1 + all_oof[names[2]]*w2
        r2 = r2_score(y, trial)
        if r2 > best_r2: best_r2, best_w = r2, [w0,w1,w2]

print(f"  Optimal weights: {best_w}")
oof_final = sum(all_oof[n]*w for n,w in zip(names, best_w))
test_final = sum(all_test[n]*w for n,w in zip(names, best_w))

# Scaling
best_s, best_sr2 = 1.0, best_r2
for s in np.arange(0.8, 1.2, 0.01):
    sr2 = r2_score(y, oof_final*s)
    if sr2 > best_sr2: best_sr2, best_s = sr2, s
if best_s != 1.0:
    test_final *= best_s
    print(f"  Scaled by {best_s:.2f}")

print(f"\n  FINAL R2: {best_sr2:.6f}")

# ─── 7. Submission ──────────────────────────────────────────────────────────
sub = pd.DataFrame({"ID": test_ids, "TARGET": test_final})
sub.to_csv("submission.csv", index=False)
print(f"\nSaved submission.csv ({len(sub)} rows)")
print(sub.head())