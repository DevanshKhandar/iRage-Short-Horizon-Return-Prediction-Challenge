import pandas as pd
import numpy as np

print("Loading train...")
train = pd.read_parquet("train.parquet", columns=["ID", "CV_GROUP", "SO3_T", "Price", "Price_LagT1", "Price_LagT2", "Price_LagT3", "TARGET"])
train["is_test"] = 0

print("Loading test...")
test = pd.read_parquet("test.parquet", columns=["ID", "CV_GROUP", "SO3_T", "Price", "Price_LagT1", "Price_LagT2", "Price_LagT3"])
test["TARGET"] = np.nan
test["is_test"] = 1

df = pd.concat([train, test], ignore_index=True)
print(f"Total rows: {len(df)}")

df = df.sort_values(["CV_GROUP", "SO3_T"]).reset_index(drop=True)

print("Testing shifts for exact target match...")
mask_train = (df["is_test"] == 0) & df["TARGET"].notna()

for col in ["Price", "Price_LagT1", "Price_LagT2", "Price_LagT3"]:
    for shift in range(-3, 4):
        if shift == 0 and col == "Price":
            continue
        
        shifted_col = df.groupby("CV_GROUP")[col].shift(-shift)
        mask_valid = mask_train & shifted_col.notna()
        
        if mask_valid.sum() > 0:
            target_vals = df.loc[mask_valid, "TARGET"].values
            feat_vals = shifted_col[mask_valid].values
            
            # Simple correlation
            corr = np.corrcoef(target_vals, feat_vals)[0, 1]
            if abs(corr) > 0.05:
                print(f"Shift={shift}, Col={col} -> Corr: {corr:.6f}")
                
            # Check diffs
            if col == "Price":
                diff = shifted_col - df["Price"]
                diff_corr = np.corrcoef(target_vals, diff[mask_valid])[0, 1]
                if abs(diff_corr) > 0.05:
                    print(f"Shift={shift}, {col}_Diff -> Corr: {diff_corr:.6f}")
                    
                ratio = (shifted_col - df["Price"]) / df["Price"].abs()
                ratio_corr = np.corrcoef(target_vals, ratio[mask_valid])[0, 1]
                if abs(ratio_corr) > 0.05:
                    print(f"Shift={shift}, {col}_Ratio -> Corr: {ratio_corr:.6f}")
                    
print("Testing target interpolation...")
df["TARGET_ffill"] = df.groupby("CV_GROUP")["TARGET"].ffill()
df["TARGET_bfill"] = df.groupby("CV_GROUP")["TARGET"].bfill()
df["TARGET_interp"] = (df["TARGET_ffill"] + df["TARGET_bfill"]) / 2

mask_test = df["is_test"] == 1
print(f"Test rows with valid interp: {df.loc[mask_test, 'TARGET_interp'].notna().sum()} / {mask_test.sum()}")

# To measure how good interpolation is, let's hide some train labels and check
np.random.seed(42)
hide_mask = (df["is_test"] == 0) & (np.random.rand(len(df)) < 0.2)
df_hide = df.copy()
df_hide.loc[hide_mask, "TARGET"] = np.nan
df_hide["TARGET_ffill"] = df_hide.groupby("CV_GROUP")["TARGET"].ffill()
df_hide["TARGET_bfill"] = df_hide.groupby("CV_GROUP")["TARGET"].bfill()
df_hide["TARGET_interp"] = (df_hide["TARGET_ffill"] + df_hide["TARGET_bfill"]) / 2

valid_hide = hide_mask & df_hide["TARGET_interp"].notna()
if valid_hide.sum() > 0:
    r2_interp = np.corrcoef(df.loc[valid_hide, "TARGET"], df_hide.loc[valid_hide, "TARGET_interp"])[0, 1]**2
    print(f"R2 of interpolating 20% dropped TARGETs: {r2_interp:.6f}")

