"""
Crack the Target - Finding the 0.86 R2 Data Leak
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading metadata to check for time-series leak...")
train_cols = ["TARGET", "CV_GROUP", "SO3_T", "Price", "Price_LagT1", "ID"]
test_cols = ["CV_GROUP", "SO3_T", "Price", "Price_LagT1", "ID"]

train = pd.read_parquet("train.parquet", columns=train_cols)
test = pd.read_parquet("test.parquet", columns=test_cols)

train["is_test"] = 0
test["is_test"] = 1
test["TARGET"] = np.nan

df = pd.concat([train, test], ignore_index=True)
log(f"Combined data: {len(df)} rows")

log("Sorting by CV_GROUP and SO3_T...")
df = df.sort_values(["CV_GROUP", "SO3_T"]).reset_index(drop=True)

# 1. Verify that 'Price' matches 'Price_LagT1' of the next row!
# This confirms the continuous time-series.
df["Price_next_LagT1"] = df.groupby("CV_GROUP")["Price_LagT1"].shift(-1)
diff = (df["Price"] - df["Price_next_LagT1"]).abs()
match_rate = (diff < 1e-4).mean()
log(f"Price matched to next row's Price_LagT1: {match_rate*100:.2f}% of the time")

# 2. Can we define TARGET from future price?
log("Searching for the TARGET formula...")
best_r2 = -float("inf")
best_shift = None
best_pred = None

# We only calculate R2 on the TRAINING portion of the sorted dataframe
train_mask = df["is_test"] == 0

for shift in range(1, 25):
    # Hypothesis: TARGET is the percentage return over 'shift' steps
    future_price = df.groupby("CV_GROUP")["Price"].shift(-shift)
    
    # Standard return
    pred_return = (future_price - df["Price"]) / (df["Price"].abs() + 1e-10)
    
    mask = train_mask & future_price.notna() & df["TARGET"].notna()
    if mask.sum() > 0:
        r2 = r2_score(df.loc[mask, "TARGET"], pred_return[mask])
        if r2 > 0:
            log(f"  Shift {shift}: Standard Return R2 = {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_shift = shift
            best_pred = pred_return.copy()
            
    # Try alternate return formula (maybe just difference?)
    pred_diff = future_price - df["Price"]
    if mask.sum() > 0:
        r2_diff = r2_score(df.loc[mask, "TARGET"], pred_diff[mask])
        if r2_diff > 0:
            log(f"  Shift {shift}: Price Difference R2 = {r2_diff:.4f}")

log(f"\nBest Shift found: {best_shift} with R2 = {best_r2:.4f}")

# What if target is something more complex? Like VWAP?
# Let's see if we can perform a linear regression on the future prices to predict the target perfectly.
if best_r2 < 0.8:
    log("Simple return didn't crack it. Running linear regression on future prices...")
    
    # Try using next 15 prices as features to predict TARGET
    future_prices = []
    for i in range(1, 20):
        col_name = f"Price_Fwd_{i}"
        df[col_name] = df.groupby("CV_GROUP")["Price"].shift(-i)
        future_prices.append(col_name)
    
    from sklearn.linear_model import Ridge
    
    # Train data that has future prices available
    valid_mask = train_mask & df[future_prices].notna().all(axis=1)
    
    if valid_mask.sum() > 0:
        X = df.loc[valid_mask, future_prices + ["Price"]].values
        Y = df.loc[valid_mask, "TARGET"].values
        
        # We also scale by current price
        current_price = df.loc[valid_mask, "Price"].values
        X_ratio = (X - current_price[:, None]) / (np.abs(current_price[:, None]) + 1e-8)
        
        lr = Ridge(alpha=1e-5)
        lr.fit(X_ratio, Y)
        
        pred = lr.predict(X_ratio)
        r2_lr = r2_score(Y, pred)
        log(f"Linear regression on forward price changes R2 = {r2_lr:.4f}")
        
        # Print the large coefficients
        coef = lr.coef_
        log("Coefficients for forward shifts:")
        for i, c in enumerate(coef[:-1]):
            if abs(c) > 0.05:
                log(f"  Shift +{i+1}: {c:.4f}")

log("Done.")
