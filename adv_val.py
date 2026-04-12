import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time

print("Loading data for adversarial validation...")
# Load a subset of rows to save time
meta_train = pd.read_parquet("train.parquet", columns=["ID"]).head(200000)
meta_test = pd.read_parquet("test.parquet", columns=["ID"]).head(200000)

features = [c for c in pq.ParquetFile("train.parquet").schema.names if c not in {"ID", "CV_GROUP", "TARGET", "SO3_T"}]
# Just use top 50 features for speed
features = features[:50]

X_train = pd.read_parquet("train.parquet", columns=features).head(200000)
X_test = pd.read_parquet("test.parquet", columns=features).head(200000)

X_train["is_test"] = 0
X_test["is_test"] = 1

df = pd.concat([X_train, X_test], axis=0)
y = df.pop("is_test")
X = df.values.astype(np.float32)

xtr, xva, ytr, yva = train_test_split(X, y.values, test_size=0.2, random_state=42, stratify=y.values)
print("Training adversarial classifier...")
bst = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
bst.fit(xtr, ytr, eval_set=[(xva, yva)], callbacks=[lgb.early_stopping(10)])

preds = bst.predict_proba(xva)[:, 1]
auc = roc_auc_score(yva, preds)
print(f"Adversarial Validation AUC: {auc:.4f}")

if auc > 0.8:
    print("WARNING: Train and Test distributions are heavily mismatched!")
    importances = bst.feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    print("Top skewed features:")
    for i in idx:
        print(f"  {features[i]}: {importances[i]}")
else:
    print("Train and Test distributions are relatively similar.")
