"""
Deep feature analysis for the competition.
Understanding the data structure and potential feature interactions.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import gc

# KEY INSIGHT: Price_LagT1 mean is very close to 0 with std ~0.033
# This means the Lag features are DIFFERENCES not lagged values
# TARGET has similar scale (std ~0.037) - TARGET is the NEXT return

# Load meta
meta = pd.read_parquet('train.parquet', columns=['TARGET','CV_GROUP','ID'])
y = meta['TARGET'].values.astype(np.float32)
groups = meta['CV_GROUP'].values
n = len(y)

# CRITICAL TEST 1: Check if there are interactions between non-lag features
# that predict target much better than individual features
print("=== Test 1: Non-lag feature interactions ===")
non_lag_important = ['S01_O02', 'S01_O02_A01', 'S02_O01', 'S02_O02', 'S02_O02_A01', 
                     'S02_O01_A01', 'S01_O01', 'S01_O01_A01', 'S03_V14_I01',
                     'S02_F03_U01', 'Price', 'S04_V19_A06', 'SO3_T']
df = pd.read_parquet('train.parquet', columns=non_lag_important + ['TARGET', 'CV_GROUP'])

# Add interaction features
df['S01_O02_x_S02_O02'] = df['S01_O02'] * df['S02_O02']
df['S01_O01_x_S02_O01'] = df['S01_O01'] * df['S02_O01']
df['O_ratio'] = df['S01_O02'] / (df['S02_O02'].abs() + 1e-10)
df['O_diff'] = df['S01_O02'] - df['S02_O02']
df['O_sum'] = df['S01_O02'] + df['S02_O02']

interaction_features = ['S01_O02_x_S02_O02', 'S01_O01_x_S02_O01', 'O_ratio', 'O_diff', 'O_sum']
for c in interaction_features:
    print(f'  {c} corr with TARGET: {df[c].corr(df["TARGET"]):.6f}')

# CRITICAL TEST 2: Simple LightGBM with JUST non-lag features
print()
print("=== Test 2: LightGBM with non-lag features only ===")
X_nonlag = df[non_lag_important].values.astype(np.float32)
gkf = GroupKFold(n_splits=5)
oof = np.zeros(n)
for fi, (tr, va) in enumerate(gkf.split(X_nonlag, y, groups)):
    dt = lgb.Dataset(X_nonlag[tr], y[tr], feature_name=non_lag_important, free_raw_data=True)
    dv = lgb.Dataset(X_nonlag[va], y[va], feature_name=non_lag_important, free_raw_data=True, reference=dt)
    b = lgb.train(
        {"objective":"regression", "metric":"mse", "learning_rate":0.01, "num_leaves":31,
         "max_depth":6, "feature_fraction":0.8, "bagging_fraction":0.8, "bagging_freq":1,
         "reg_alpha":0.1, "reg_lambda":1.0, "verbose":-1},
        dt, 5000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof[va] = b.predict(X_nonlag[va])
    del dt, dv, b; gc.collect()
print(f'  Non-lag features only R2: {r2_score(y, oof):.6f}')

# CRITICAL TEST 3: ALL features (no feature selection, just dump everything)
print()
print("=== Test 3: LightGBM with ALL features, no selection ===")
del df; gc.collect()

import pyarrow.parquet as pq
all_cols = pq.ParquetFile('train.parquet').schema.names
feat_cols = [c for c in all_cols if c not in {'ID','CV_GROUP','TARGET'}]

# Load all features
X_all = pd.read_parquet('train.parquet', columns=feat_cols).values.astype(np.float32)
print(f'  Shape: {X_all.shape}')

oof2 = np.zeros(n)
for fi, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
    print(f'  Fold {fi+1}...')
    dt = lgb.Dataset(X_all[tr], y[tr], feature_name=feat_cols, free_raw_data=True)
    dv = lgb.Dataset(X_all[va], y[va], feature_name=feat_cols, free_raw_data=True, reference=dt)
    b = lgb.train(
        {"objective":"regression", "metric":"mse", "learning_rate":0.01, "num_leaves":63,
         "max_depth":7, "feature_fraction":0.5, "bagging_fraction":0.7, "bagging_freq":1,
         "reg_alpha":0.5, "reg_lambda":2.0, "min_child_samples":100, "verbose":-1},
        dt, 5000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof2[va] = b.predict(X_all[va])
    print(f'    Best iter: {b.best_iteration}, val R2: {r2_score(y[va], oof2[va]):.6f}')
    del dt, dv, b; gc.collect()
print(f'  All features R2: {r2_score(y, oof2):.6f}')

# CRITICAL TEST 4: LightGBM with dart boosting
print()
print("=== Test 4: LightGBM with DART boosting ===")
oof3 = np.zeros(n)
for fi, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
    print(f'  Fold {fi+1}...')
    dt = lgb.Dataset(X_all[tr], y[tr], feature_name=feat_cols, free_raw_data=True)
    dv = lgb.Dataset(X_all[va], y[va], feature_name=feat_cols, free_raw_data=True, reference=dt)
    b = lgb.train(
        {"objective":"regression", "metric":"mse", "boosting_type":"dart", "learning_rate":0.01,
         "num_leaves":31, "max_depth":6, "feature_fraction":0.5, "bagging_fraction":0.7,
         "bagging_freq":1, "reg_alpha":0.5, "reg_lambda":2.0, "min_child_samples":200,
         "drop_rate":0.1, "max_drop":50, "verbose":-1},
        dt, 500, valid_sets=[dv], callbacks=[lgb.log_evaluation(100)])
    oof3[va] = b.predict(X_all[va])
    print(f'    val R2: {r2_score(y[va], oof3[va]):.6f}')
    del dt, dv, b; gc.collect()
print(f'  DART R2: {r2_score(y, oof3):.6f}')

# CRITICAL TEST 5: CatBoost comparison
print()
print("=== Test 5: Trying XGBoost ===")
try:
    import xgboost as xgb
    oof4 = np.zeros(n)
    for fi, (tr, va) in enumerate(gkf.split(X_all, y, groups)):
        print(f'  Fold {fi+1}...')
        dtrain = xgb.DMatrix(X_all[tr], label=y[tr], feature_names=feat_cols)
        dval = xgb.DMatrix(X_all[va], label=y[va], feature_names=feat_cols)
        b = xgb.train(
            {"objective":"reg:squarederror", "eval_metric":"rmse", "learning_rate":0.01,
             "max_depth":6, "subsample":0.7, "colsample_bytree":0.5,
             "reg_alpha":0.5, "reg_lambda":2.0, "min_child_weight":200},
            dtrain, 5000, evals=[(dval,"val")], early_stopping_rounds=100, verbose_eval=0)
        oof4[va] = b.predict(dval)
        print(f'    val R2: {r2_score(y[va], oof4[va]):.6f}')
        del dtrain, dval, b; gc.collect()
    print(f'  XGBoost R2: {r2_score(y, oof4):.6f}')
except ImportError:
    print('  XGBoost not available')

del X_all; gc.collect()
print()
print("DONE!")
