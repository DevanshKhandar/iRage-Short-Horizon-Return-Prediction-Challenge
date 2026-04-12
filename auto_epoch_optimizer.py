"""
Auto Epoch Optimizer
====================
Keeps iterating and epoching LightGBM architectures endlessly 
using Optuna to extract every physical drop of R2 score from the dataset.
"""

import os, gc, time, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Phase 1: Loading Datasets...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = meta["TARGET"].values.astype(np.float32)
groups = meta["CV_GROUP"].values
n_train = len(y)
del meta; gc.collect()

all_cols = pq.ParquetFile("train.parquet").schema.names
raw_feats = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET", "SO3_T"}]
test_col_set = set(pq.ParquetFile("test.parquet").schema.names)
safe_feats = [c for c in raw_feats if c in test_col_set]

log("Phase 2: Building Massive Correlation Matrix...")
y_c = y - y.mean()
corrs = {}
for i in range(0, len(safe_feats), 50):
    b = safe_feats[i:i+50]
    df = pd.read_parquet("train.parquet", columns=b)
    for c in b:
        v = df[c].values.astype(np.float32)
        vs = v.std()
        corrs[c] = float(np.dot(y_c, v - v.mean()) / (n_train * y.std() * vs)) if vs > 0 else 0.0
    del df; gc.collect()

# Taking the top 150 features
TOP_N = 150 
selected = [name for name, _ in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:TOP_N]]
del corrs, y_c; gc.collect()

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

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

BEST_SCORE_GLOBAL = -float('inf')

log("\nPhase 3: Launching Endless Epoch Iterator...")

def objective(trial):
    global BEST_SCORE_GLOBAL
    
    # We let it dynamically iterate combinations of topologies
    params = {
        "objective": trial.suggest_categorical("objective", ["regression", "huber", "fair"]),
        "metric": "root_mean_squared_error",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": 1,
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 20.0),
        "verbose": -1,
        "n_jobs": -1,
        "random_state": trial.number
    }
    
    if params["objective"] == "huber":
        params["alpha"] = trial.suggest_float("alpha", 0.8, 0.99)
    if params["objective"] == "fair":
        params["fair_c"] = trial.suggest_float("fair_c", 0.5, 5.0)

    oof_preds = np.zeros(n_train, np.float32)
    
    for fi, (tr, va) in enumerate(gkf.split(X_train, y_clip, groups)):
        dt = lgb.Dataset(X_train[tr], y_clip[tr], free_raw_data=True)
        dv = lgb.Dataset(X_train[va], y_clip[va], free_raw_data=True, reference=dt)
        
        # Keep epoching endlessly (up to 10000) until the local valid set mathematically stops learning
        bst = lgb.train(params, dt, 10000, valid_sets=[dv], valid_names=["v"],
                        callbacks=[lgb.early_stopping(150, verbose=False),
                                   optuna.integration.LightGBMPruningCallback(trial, "rmse")])
        oof_preds[va] = bst.predict(X_train[va])
        del dt, dv, bst; gc.collect()
        
    score = r2_score(y, oof_preds)
    
    if score > BEST_SCORE_GLOBAL:
        BEST_SCORE_GLOBAL = score
        log(f"BREAKTHROUGH: New Max R2 Achieved -> {score:.7f}")
        
    return score

# Let it epoch iteratively forever until forced abort.
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=100))
study.optimize(objective, n_trials=10000)

log(f"\nFinal Global Maximum R2 after Iterator: {study.best_value:.7f}")
log(f"Best Params: {study.best_params}")
