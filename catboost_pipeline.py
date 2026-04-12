"""
CatBoost High-End Training Pipeline
===================================
CatBoost is exceptionally strong out-of-the-box for tabular data compared to LightGBM.
It uses symmetric trees which combat overfitting, a huge problem in financial datasets.
"""

import gc, time, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")
np.random.seed(42)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("PHASE 1: Loading metadata...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = meta["TARGET"].values.astype(np.float32)
groups = meta["CV_GROUP"].values
n_train = len(y)
del meta; gc.collect()

test_meta = pd.read_parquet("test.parquet", columns=["ID"])
n_test = len(test_meta)

log("PHASE 2: Loading ALL features...")
all_cols = pq.ParquetFile("train.parquet").schema.names
test_cols = set(pq.ParquetFile("test.parquet").schema.names)
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"} and c in test_cols]

# We will sample slightly to speed up training if needed, but for now use full data
X_train = np.empty((n_train, len(feat_cols)), dtype=np.float32)
for i in range(0, len(feat_cols), 50):
    batch = feat_cols[i:i+50]
    df = pd.read_parquet("train.parquet", columns=batch)
    for j, c in enumerate(batch):
        X_train[:, i+j] = df[c].values.astype(np.float32)
    del df; gc.collect()

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

log("\nPHASE 3: CatBoost Training...")
N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

oof_preds = np.zeros(n_train, np.float32)
models = []

for fi, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y, groups)):
    log(f"  Training Fold {fi+1}/{N_FOLDS}...")
    
    train_pool = Pool(X_train[tr_idx], y[tr_idx])
    val_pool = Pool(X_train[va_idx], y[va_idx])
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fi,
        od_type='Iter',
        od_wait=50, # early stopping
        task_type='CPU',
        thread_count=-1,
        verbose=100
    )
    
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_preds[va_idx] = model.predict(val_pool)
    models.append(model)
    
    del train_pool, val_pool; gc.collect()

r2_cv = r2_score(y, oof_preds)
log(f"\n>>> CatBoost OOF R2: {r2_cv:.6f} <<<")

log("\nPHASE 4: Test Prediction...")
del X_train; gc.collect()

X_test = np.empty((n_test, len(feat_cols)), dtype=np.float32)
for i in range(0, len(feat_cols), 50):
    batch = feat_cols[i:i+50]
    df = pd.read_parquet("test.parquet", columns=batch)
    for j, c in enumerate(batch):
        X_test[:, i+j] = df[c].values.astype(np.float32)
    del df; gc.collect()

X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

test_preds = np.zeros(n_test, np.float32)
for model in models:
    test_preds += model.predict(X_test) / N_FOLDS

sub = pd.DataFrame({"ID": test_meta["ID"].values, "TARGET": test_preds})
sub.to_csv("submission_catboost.csv", index=False)

log(f"\nSubmission generated with mean={test_preds.mean():.6f}, std={test_preds.std():.6f}")

# Also let's try blending CatBoost with LightGBM
try:
    lgb_sub = pd.read_csv("submission_clutch_v1.csv")
    blend = 0.5 * test_preds + 0.5 * lgb_sub["TARGET"].values
    sub_blend = pd.DataFrame({"ID": test_meta["ID"].values, "TARGET": blend})
    sub_blend.to_csv("submission_catboost_lgbm_blend.csv", index=False)
    log("Generated CatBoost + LightGBM blend!")
except Exception as e:
    log(f"Could not generate blend: {e}")

log("DONE")
