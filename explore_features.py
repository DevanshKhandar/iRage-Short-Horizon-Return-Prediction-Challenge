"""Quick feature exploration."""
import pandas as pd
import numpy as np

cols_check = ['Price', 'TARGET', 'S01_O01', 'S01_O02', 'S02_O01', 'S02_O02', 'SO3_T', 'S03_V14_I01']
df = pd.read_parquet('train.parquet', columns=cols_check)
print('=== Feature stats ===')
for c in cols_check:
    print(f'{c}: mean={df[c].mean():.6f}, std={df[c].std():.6f}, min={df[c].min():.6f}, max={df[c].max():.6f}')

print()
print('=== Correlations with TARGET ===')
for c in [x for x in cols_check if x != 'TARGET']:
    print(f'  {c}: {df[c].corr(df["TARGET"]):.6f}')

# Check if TARGET = (next Price - Price) / Price for any lag structure
lag_cols = ['Price_LagT1', 'Price_LagT2', 'Price_LagT3']
df2 = pd.read_parquet('train.parquet', columns=['Price', 'TARGET'] + lag_cols)
print()
print('=== Price vs Price lags ===')
for c in lag_cols:
    print(f'{c}: mean={df2[c].mean():.6f}, corr with Price={df2[c].corr(df2["Price"]):.6f}')

# Check if TARGET ~= (Price - PriceLagT1) / PriceLagT1 or similar
ret1 = (df2['Price'] - df2['Price_LagT1']) / df2['Price_LagT1'].abs().clip(lower=1e-10)
ret2 = df2['Price'] - df2['Price_LagT1']
print()
print('=== Constructed returns ===')
print(f'(Price-LagT1)/LagT1 corr with TARGET: {ret1.corr(df2["TARGET"]):.6f}')
print(f'(Price-LagT1) corr with TARGET: {ret2.corr(df2["TARGET"]):.6f}')
print(f'Price corr with TARGET: {df2["Price"].corr(df2["TARGET"]):.6f}')

# Check non-zero fraction
print()
print(f'TARGET zero fraction: {(df2["TARGET"]==0).mean():.4f}')
print(f'TARGET near-zero (abs<1e-6): {(df2["TARGET"].abs()<1e-6).mean():.4f}')

# Check for potential leaking features
print()
print('=== Checking non-lag features for high R2 ===')
all_non_lag = ['S01_F01_U01', 'S01_F02_U01', 'S01_F03_U01', 'S02_F01_U01', 'S02_F02_U01', 'S02_F03_U01',
               'S01_O01', 'S01_O02', 'S01_O01_A01', 'S01_O02_A01', 'S02_O01', 'S02_O02', 
               'S02_O01_A01', 'S02_O02_A01', 'S04_V19_V12', 'S04_V19_A06', 'S05_V19', 'SO3_T', 'Price']
df3 = pd.read_parquet('train.parquet', columns=all_non_lag + ['TARGET'])
print('Correlations:')
for c in all_non_lag:
    corr = df3[c].corr(df3['TARGET'])
    if abs(corr) > 0.005:
        print(f'  ** {c}: {corr:+.6f}')
    else:
        print(f'     {c}: {corr:+.6f}')
