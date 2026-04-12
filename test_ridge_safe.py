import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pyarrow.parquet as pq

# Load specific safe columns to test
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID"])
y = meta["TARGET"].values.astype(np.float32)
groups = meta["CV_GROUP"].values

all_cols = pq.ParquetFile("train.parquet").schema.names
raw_feats = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET", "SO3_T"}]

# Load a quick subset of 50 features
X = pd.read_parquet("train.parquet", columns=raw_feats[:50]).fillna(0).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(y))

for tr, va in gkf.split(X_scaled, y, groups):
    model = Ridge(alpha=100.0)
    model.fit(X_scaled[tr], y[tr])
    oof[va] = model.predict(X_scaled[va])

print(f"Ridge OOF R2: {r2_score(y, oof):.6f}")
