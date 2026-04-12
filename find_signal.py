"""
Deep signal hunting: What relationship gives R2=0.86?
If the leaderboard top is 0.86, there's a near-deterministic signal.
Let's find it.
"""
import gc, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import r2_score

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Load target
log("Loading data...")
meta = pd.read_parquet("train.parquet", columns=["TARGET", "CV_GROUP", "ID", "Price", 
    "Price_LagT1", "Price_LagT2", "Price_LagT3", "SO3_T"])
y = meta["TARGET"].values
n = len(y)

log(f"n={n}, target: mean={y.mean():.8f}, std={y.std():.8f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 1: Is TARGET exactly some function of Price?
# ══════════════════════════════════════════════════════════════════════
log("\n=== TEST 1: Price-based hypotheses ===")

price = meta["Price"].values
pl1 = meta["Price_LagT1"].values
pl2 = meta["Price_LagT2"].values
pl3 = meta["Price_LagT3"].values

tests = {
    "Price_LagT1 (raw)": pl1,
    "Price_LagT2": pl2,
    "Price_LagT3": pl3,
    "Price": price,
    "(Price-PL1)/PL1": (price - pl1) / (np.abs(pl1) + 1e-10),
    "PL1/Price": pl1 / (np.abs(price) + 1e-10),
    "Price-PL1": price - pl1,
    "PL1-PL2": pl1 - pl2,
    "PL1+PL2+PL3": pl1 + pl2 + pl3,
}
for name, pred in tests.items():
    mask = np.isfinite(pred)
    if mask.sum() > 0:
        r2 = r2_score(y[mask], pred[mask])
        corr = np.corrcoef(y[mask], pred[mask])[0,1]
        log(f"  {name:30s}: R2={r2:+.6f}, corr={corr:+.6f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 2: Check every single feature for high R2
# ══════════════════════════════════════════════════════════════════════
log("\n=== TEST 2: Individual feature R2 (not just correlation) ===")

all_cols = pq.ParquetFile("train.parquet").schema.names
feat_cols = [c for c in all_cols if c not in {"ID", "CV_GROUP", "TARGET"}]

top_r2 = []
for i, c in enumerate(feat_cols):
    v = pd.read_parquet("train.parquet", columns=[c])[c].values.astype(np.float64)
    mask = np.isfinite(v)
    if mask.sum() < n * 0.9:
        continue
    
    # R2 if we use the feature directly as prediction
    r2_raw = r2_score(y[mask], v[mask])
    
    # R2 with optimal linear scaling: y = a*v + b
    # Optimal a = cov(y,v)/var(v), b = mean(y) - a*mean(v)
    vm, ym = v[mask].mean(), y[mask].mean()
    vvar = ((v[mask] - vm)**2).mean()
    if vvar > 0:
        a = ((y[mask] - ym) * (v[mask] - vm)).mean() / vvar
        b = ym - a * vm
        pred_lin = a * v[mask] + b
        r2_lin = r2_score(y[mask], pred_lin)
    else:
        r2_lin = 0.0
    
    top_r2.append((c, r2_raw, r2_lin))
    del v
    if (i+1) % 100 == 0:
        log(f"  {i+1}/{len(feat_cols)}")
        gc.collect()

# Sort by linear R2
top_r2.sort(key=lambda x: x[2], reverse=True)
log("\n  TOP 20 features by linear R2:")
for name, r2_raw, r2_lin in top_r2[:20]:
    log(f"    {name:45s}: raw_R2={r2_raw:+.6f}, linear_R2={r2_lin:+.6f}")

log("\n  BOTTOM 5 (most negative raw R2):")
top_r2.sort(key=lambda x: x[1])
for name, r2_raw, r2_lin in top_r2[:5]:
    log(f"    {name:45s}: raw_R2={r2_raw:+.6f}, linear_R2={r2_lin:+.6f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 3: Check if TARGET depends on CV_GROUP
# Maybe time-dependent signal?
# ══════════════════════════════════════════════════════════════════════
log("\n=== TEST 3: Per-group analysis ===")
groups = meta["CV_GROUP"].values
for g in sorted(np.unique(groups)):
    mask = groups == g
    gn = mask.sum()
    gm = y[mask].mean()
    gs = y[mask].std()
    log(f"  Group {g:2d}: n={gn:6d}, mean={gm:+.8f}, std={gs:.6f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 4: Check SO3_T (potential timestamp) ordering
# Maybe within each group, features predict better?
# ══════════════════════════════════════════════════════════════════════
log("\n=== TEST 4: Within-group prediction ===")
so3t = meta["SO3_T"].values
log(f"  SO3_T: min={so3t.min()}, max={so3t.max()}, nunique={len(np.unique(so3t))}")

# Sort by SO3_T within each group and test if lag features predict better
# This tests if the data has temporal structure we can exploit
log("  Testing if sorted data helps prediction...")
sorted_idx = np.lexsort((so3t, groups))
y_sorted = y[sorted_idx]
pl1_sorted = pl1[sorted_idx]

# Simple test: does next row's target == current Price change?
log(f"  Correlation of consecutive targets: {np.corrcoef(y_sorted[:-1], y_sorted[1:])[0,1]:.6f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 5: Feature combinations - pairs of top features
# ══════════════════════════════════════════════════════════════════════
log("\n=== TEST 5: Top feature pairs ===")
top_feat_names = [name for name, _, r2l in sorted(top_r2, key=lambda x: x[2], reverse=True)[:10]]
log(f"  Using top 10 features: {top_feat_names}")

# Load top features
top_data = pd.read_parquet("train.parquet", columns=top_feat_names + ["TARGET"])

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Try pairs
best_pair_r2 = 0
best_pair = None
for i, f1 in enumerate(top_feat_names):
    for j, f2 in enumerate(top_feat_names):
        if j <= i: continue
        X = np.column_stack([top_data[f1].values, top_data[f2].values]).astype(np.float64)
        mask = np.all(np.isfinite(X), axis=1)
        if mask.sum() < n * 0.9: continue
        
        vm = X[mask].mean(axis=0)
        vs = X[mask].std(axis=0)
        vs[vs == 0] = 1
        Xs = (X[mask] - vm) / vs
        
        # Simple OLS
        XtX = Xs.T @ Xs
        Xty = Xs.T @ y[mask]
        try:
            beta = np.linalg.solve(XtX, Xty)
            pred = Xs @ beta
            r2 = r2_score(y[mask], pred)
            if r2 > best_pair_r2:
                best_pair_r2 = r2
                best_pair = (f1, f2, r2)
                log(f"    {f1} + {f2}: R2={r2:.6f}")
        except:
            pass

log(f"\n  Best pair: {best_pair}")

# Try ALL top 10 together
log("\n  All top 10 features together (OLS):")
X_all = top_data[top_feat_names].values.astype(np.float64)
mask = np.all(np.isfinite(X_all), axis=1)
vm = X_all[mask].mean(axis=0)
vs = X_all[mask].std(axis=0); vs[vs==0] = 1
Xs = (X_all[mask] - vm) / vs
XtX = Xs.T @ Xs + 0.01 * np.eye(len(top_feat_names))
Xty = Xs.T @ y[mask]
beta = np.linalg.solve(XtX, Xty)
pred = Xs @ beta
r2 = r2_score(y[mask], pred)
log(f"  R2 = {r2:.6f}")

# ══════════════════════════════════════════════════════════════════════
# TEST 6: Check if base (non-lag) features are more predictive
# ══════════════════════════════════════════════════════════════════════
log("\n=== TEST 6: Base features only (no lags) ===")
base_cols = [c for c in feat_cols if "_Lag" not in c]
log(f"  {len(base_cols)} base features")

base_data = pd.read_parquet("train.parquet", columns=base_cols)
X_base = base_data.values.astype(np.float64)
X_base = np.nan_to_num(X_base, nan=0, posinf=0, neginf=0)
del base_data; gc.collect()

vm = X_base.mean(axis=0); vs = X_base.std(axis=0); vs[vs==0] = 1
Xs = (X_base - vm) / vs

# Ridge with various alphas on base features only
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for alpha in [1, 10, 100, 1000, 10000, 100000]:
    oof = np.zeros(n)
    for tr, va in gkf.split(Xs, y, groups):
        ridge = Ridge(alpha=alpha)
        ridge.fit(Xs[tr], y[tr])
        oof[va] = ridge.predict(Xs[va])
    r2 = r2_score(y, oof)
    log(f"  Ridge alpha={alpha:>7}: R2={r2:.6f}")

del X_base, Xs; gc.collect()

log("\n=== INVESTIGATION COMPLETE ===")
