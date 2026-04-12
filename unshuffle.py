import pandas as pd
import numpy as np
from collections import defaultdict
import time

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading chunks...")
cols = ['ID', 'Price', 'Price_LagT1', 'Price_LagT2', 'Price_LagT3']
train = pd.read_parquet('train.parquet', columns=cols + ['TARGET'])
train['is_test'] = 0
test = pd.read_parquet('test.parquet', columns=cols)
test['TARGET'] = np.nan
test['is_test'] = 1

df = pd.concat([train, test], ignore_index=True)
log(f"Length: {len(df)}")

log("Building maps...")
d1 = np.round(df['Price_LagT1'].values, 4)
d2 = np.round(df['Price_LagT2'].values, 4)
d3 = np.round(df['Price_LagT3'].values, 4)

# B's identity is defined by its Lag2 and Lag3 matching A's Lag1 and Lag2
b_map = defaultdict(list)
for i in range(len(df)):
    k = (d2[i], d3[i])
    b_map[k].append(i)

log("Hunting linkages in first 200k train rows...")
results = []
matches = 0

for i in range(200000):
    if df['is_test'].values[i] == 1: continue
        
    k_query = (d1[i], d2[i])
    if k_query in b_map:
        cands = b_map[k_query]
        valid_b = None
        for b_idx in cands:
            diff1 = abs(df['Price_LagT2'].values[b_idx] - df['Price_LagT1'].values[i])
            diff2 = abs(df['Price_LagT3'].values[b_idx] - df['Price_LagT2'].values[i])
            if diff1 < 1e-4 and diff2 < 1e-4:
                valid_b = b_idx
                break
                
        if valid_b is not None:
            matches += 1
            results.append({
                'A_ID': df['ID'].values[i],
                'TARGET_A': df['TARGET'].values[i],
                'Lag1_B': df['Price_LagT1'].values[valid_b],
                'Price_A': df['Price'].values[i],
                'Price_B': df['Price'].values[valid_b],
                'TARGET_B': df['TARGET'].values[valid_b]
            })

log(f"Total Matches found: {matches}")
if matches > 0:
    res = pd.DataFrame(results)
    
    calc = res['Lag1_B'] / res['Price_A']
    calc2 = (res['Price_B'] - res['Price_A']) / res['Price_A']
    calc3 = res['Lag1_B']
    
    r1 = np.corrcoef(res['TARGET_A'].fillna(0), calc.fillna(0))[0,1]
    r2 = np.corrcoef(res['TARGET_A'].fillna(0), calc2.fillna(0))[0,1]
    r3 = np.corrcoef(res['TARGET_A'].fillna(0), calc3.fillna(0))[0,1]
    
    log(f"Corr(TARGET_A, Lag1_B / Price_A): {r1:.6f}")
    log(f"Corr(TARGET_A, (Price_B - Price_A) / Price_A): {r2:.6f}")
    log(f"Corr(TARGET_A, Lag1_B): {r3:.6f}")

    print("\nSAMPLE ROWS:")
    print(res.head(10))
