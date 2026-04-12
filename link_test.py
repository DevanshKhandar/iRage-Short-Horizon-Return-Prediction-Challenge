import pandas as pd
import numpy as np

print("Loading small slice...")
train = pd.read_parquet("train.parquet", columns=["ID", "Price", "Price_LagT1", "Price_LagT2", "Price_LagT3", "TARGET"]).head(50000)

for idx in range(5):
    row0 = train.iloc[idx]
    
    target_lag1 = row0["Price_LagT1"]
    target_lag2 = row0["Price_LagT2"]
    
    matches = train[
        np.isclose(train["Price_LagT2"], target_lag1, atol=1e-5) & 
        np.isclose(train["Price_LagT3"], target_lag2, atol=1e-5)
    ]
    
    if len(matches) > 0:
        print(f"\n--- Row {idx} matches ---")
        print(f"TARGET(current) = {row0['TARGET']:.6f}")
        for _, match_row in matches.iterrows():
            lag1_next = match_row['Price_LagT1']
            print(f"Price_LagT1(next) = {lag1_next:.6f}")
            # Try predicting TARGET
            # Hypothesis 1: TARGET = lag1_next
            # Hypothesis 2: TARGET = lag1_next / Price(current)
            # Hypothesis 3: TARGET = lag1_next / Price(next)
            
            p_cur = row0['Price']
            p_next = match_row['Price']
            
            print(f"Hypo 1 (lag1_next): {lag1_next:.6f}")
            print(f"Hypo 2 (lag1_next / p_cur): {lag1_next / p_cur:.6f}")
            print(f"Hypo 3 (lag1_next / p_next): {lag1_next / p_next:.6f}")
            print(f"Diff 2 from TARGET: {(lag1_next / p_cur) - row0['TARGET']:.8f}")
