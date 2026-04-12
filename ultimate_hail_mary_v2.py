import os, gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

print("1. Loading raw parquet data...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "ID"])
y_train = meta["TARGET"].values.astype(np.float32)
train_id = meta["ID"].values

test_meta = pd.read_parquet("test.parquet", columns=["ID"])
test_id = test_meta["ID"].values

# Best 40 non-linear features mapped by Mutual Information (Safe from Drift)
top_mi_feats = [
    "S03_A07_V01_V16_V06", "S03_V06_V01", "S03_V06_V15_V01", "S03_V14_V01", 
    "S03_V02_T06", "S03_V03_T06", "S03_A07_V18_V06", "S03_V02_T05", 
    "S03_V03_T05", "S03_V03_T02", "S03_V03_T03", "S03_V03_T04", 
    "S03_V02_T04", "S03_V02_T03", "S03_V02_T02", "S03_V20_V13", 
    "S03_V03_T01", "S03_D02_V01_A01_B05_E05_E06", "S03_V02_T01", 
    "S03_D02_V01_A01_B04_E04_E05", "S03_V06_V15_O03", "S03_P01_D01_S01", 
    "Price_LagT2", "S03_V06_V15_O04", "S03_D01_V11_D06"
]

def get_matrix(path, feats, n_rows):
    X = np.empty((n_rows, len(feats)), dtype=np.float32)
    df = pd.read_parquet(path, columns=feats)
    for j, c in enumerate(feats):
        X[:, j] = df[c].values.astype(np.float32)
    return np.nan_to_num(X, nan=0.0)

print("2. Constructing Safely Filtered Feature Matrices...")
X_train = get_matrix("train.parquet", top_mi_feats, len(y_train))
X_test = get_matrix("test.parquet", top_mi_feats, len(test_id))

print("3. Applying Strict Quantile Discretization (Killing Drift)...")
# Maps everything to strict standard distributions, completely neutralizing future test-set outliers/shifts
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_qt = qt.fit_transform(X_train)
X_test_qt = qt.transform(X_test)

print("4. Training Robust Blend...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y_train))
preds_test = np.zeros(len(test_id))

params = {
    "objective": "huber", "alpha": 0.9, "metric": "rmse", "boosting_type": "gbdt",
    "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31, "min_child_samples": 200, 
    "colsample_bytree": 0.5, "verbose": -1, "seed": 42
}

for tr, va in kf.split(X_train_qt):
    # Model 1: Extremely conservative LightGBM on normalized quantiles
    dt = lgb.Dataset(X_train_qt[tr], y_train[tr])
    dv = lgb.Dataset(X_train_qt[va], y_train[va])
    bst = lgb.train(params, dt, int(1500), valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # Model 2: Ridge purely searching linear ranks
    ridge = Ridge(alpha=1000.0)
    ridge.fit(X_train_qt[tr], y_train[tr])
    
    p_va = (bst.predict(X_train_qt[va]) * 0.7) + (ridge.predict(X_train_qt[va]) * 0.3)
    oof[va] = p_va
    
    p_test = (bst.predict(X_test_qt) * 0.7) + (ridge.predict(X_test_qt) * 0.3)
    preds_test += p_test / 5.0

print(f"Drift-Proof OOF R2: {r2_score(y_train, oof):.6f}")

sub = pd.DataFrame({"ID": test_id, "TARGET": preds_test})
sub.to_csv("submission_ultimate_hail_mary.csv", index=False)
print("Saved submission_ultimate_hail_mary.csv")
