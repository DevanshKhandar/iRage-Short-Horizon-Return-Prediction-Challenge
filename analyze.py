"""Memory-efficient analysis - processes columns one at a time."""
import gc
import numpy as np
import pandas as pd

# Load only needed columns for correlation analysis
print('Loading target and feature correlations...')
train = pd.read_parquet('train.parquet', columns=['TARGET', 'CV_GROUP', 'ID'])
y = train['TARGET'].values.astype(np.float32)
groups = train['CV_GROUP'].values
n = len(y)
print(f'  Loaded {n} rows')

# Get all column names
import pyarrow.parquet as pq
pf = pq.ParquetFile('train.parquet')
all_cols = pf.schema.names
drop = {'ID', 'CV_GROUP', 'TARGET'}
feat_cols = [c for c in all_cols if c not in drop]
print(f'  {len(feat_cols)} features')

# Compute correlations one column at a time
print('\nComputing correlations...')
y_mean = y.mean()
y_std = y.std()
y_centered = y - y_mean

corrs = {}
for i, c in enumerate(feat_cols):
    col = pd.read_parquet('train.parquet', columns=[c])[c].values.astype(np.float32)
    col_mean = col.mean()
    col_std = col.std()
    if col_std > 0:
        corrs[c] = np.dot(y_centered, col - col_mean) / (n * y_std * col_std)
    else:
        corrs[c] = 0.0
    del col
    if (i+1) % 50 == 0:
        print(f'  {i+1}/{len(feat_cols)}')
        gc.collect()

# Sort and display
corrs_sorted = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
print('\n=== TOP 50 FEATURES BY |CORRELATION| WITH TARGET ===')
for name, corr in corrs_sorted[:50]:
    print(f'  {corr:+.6f}  {name}')

abs_corrs = [abs(v) for v in corrs.values()]
print(f'\n  Max |corr|: {max(abs_corrs):.6f}')
print(f'  Mean |corr|: {np.mean(abs_corrs):.6f}')
print(f'  Features |corr| > 0.01: {sum(1 for c in abs_corrs if c > 0.01)}')
print(f'  Features |corr| > 0.005: {sum(1 for c in abs_corrs if c > 0.005)}')
print(f'  Features |corr| > 0.002: {sum(1 for c in abs_corrs if c > 0.002)}')

# Save sorted correlations
import json
with open('output/feature_correlations.json', 'w') as f:
    json.dump(corrs_sorted, f)
print('\nSaved feature_correlations.json')

# Target stats
from scipy import stats as sp_stats
print('\n=== TARGET DISTRIBUTION ===')
print(f'  Mean: {y.mean():.6f}')
print(f'  Std: {y.std():.6f}')
print(f'  Skew: {sp_stats.skew(y):.4f}')
print(f'  Kurtosis: {sp_stats.kurtosis(y):.4f}')
for p in [0.1, 1, 5, 95, 99, 99.9]:
    print(f'  P{p}: {np.percentile(y, p):.6f}')

# Per-group analysis
print('\n=== CV_GROUP TARGET STATS ===')
grp_means = {}
for g in np.unique(groups):
    mask = groups == g
    grp_means[int(g)] = float(y[mask].mean())
print(f'  Group means range: {min(grp_means.values()):.6f} to {max(grp_means.values()):.6f}')
print(f'  Group means std: {np.std(list(grp_means.values())):.6f}')

# Winsorization analysis
print('\n=== TARGET WINSORIZATION ===')
for pct in [99.9, 99.5, 99.0, 98.0, 97.0, 95.0]:
    lo = np.percentile(y, 100-pct)
    hi = np.percentile(y, pct)
    y_clip = np.clip(y, lo, hi)
    print(f'  Clip P{100-pct:.1f}/P{pct:.1f} [{lo:.4f},{hi:.4f}]: removed={((y<lo)|(y>hi)).mean()*100:.2f}%')

del train
gc.collect()

# Ridge regression baseline
print('\n=== RIDGE REGRESSION BASELINES ===')
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load top-correlated features only
top_n_feats = [name for name, _ in corrs_sorted[:80]]
print(f'Loading top {len(top_n_feats)} features...')
train_sub = pd.read_parquet('train.parquet', columns=top_n_feats + ['TARGET', 'CV_GROUP'])
X = train_sub[top_n_feats].values.astype(np.float32)
y2 = train_sub['TARGET'].values.astype(np.float32)
groups2 = train_sub['CV_GROUP'].values
del train_sub; gc.collect()

gkf = GroupKFold(n_splits=5)

for alpha in [0.1, 1.0, 10, 50, 100, 500, 1000, 5000, 10000]:
    oof = np.zeros(len(y2))
    for _, (tr, val) in enumerate(gkf.split(X, y2, groups2)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_val = scaler.transform(X[val])
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, y2[tr])
        oof[val] = ridge.predict(X_val)
    print(f'  Ridge alpha={alpha:>8}: R2 = {r2_score(y2, oof):.6f}')

# Also try with target clipping
print('\n=== RIDGE WITH WINSORIZED TARGET ===')
lo99 = np.percentile(y2, 1)
hi99 = np.percentile(y2, 99)
y_clip = np.clip(y2, lo99, hi99)
for alpha in [10, 100, 1000]:
    oof = np.zeros(len(y2))
    for _, (tr, val) in enumerate(gkf.split(X, y_clip, groups2)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_val = scaler.transform(X[val])
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, y_clip[tr])
        oof[val] = ridge.predict(X_val)
    # Evaluate on ORIGINAL target
    print(f'  Ridge alpha={alpha:>8} (trained on clipped): R2 = {r2_score(y2, oof):.6f}')

# ElasticNet
print('\n=== ELASTICNET ===')
from sklearn.linear_model import ElasticNet
for alpha in [0.0001, 0.001, 0.01]:
    oof = np.zeros(len(y2))
    for _, (tr, val) in enumerate(gkf.split(X, y2, groups2)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_val = scaler.transform(X[val])
        en = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=1000)
        en.fit(X_tr, y2[tr])
        oof[val] = en.predict(X_val)
    print(f'  EN alpha={alpha}: R2 = {r2_score(y2, oof):.6f}')

print('\nDone!')
