import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import gc
from collections import defaultdict

print("=====================================================")
print("  TOP 3 MASTER GAMBLE: Math Core + 445-Feat Residuals ")
print("=====================================================")

print("1. Loading timeline arrays...")
cols = ['ID', 'Price', 'Price_LagT1']
base_feats = ['S01_O02', 'S03_V04_T01', 'S02_F03_U01']
lag_feats = [f + '_LagT1' for f in base_feats]
small_cols = cols + base_feats + lag_feats

y_train = pd.read_parquet('train.parquet', columns=['TARGET'])['TARGET'].values
df_tr = pd.read_parquet('train.parquet', columns=small_cols)
df_te = pd.read_parquet('test.parquet', columns=small_cols)
df = pd.concat([df_tr, df_te], ignore_index=True)

test_ids = df_te['ID'].values
n_tr = len(df_tr)
del df_tr, df_te; gc.collect()

def get_curr(row): return (np.round(row[1], 4), np.round(row[3], 4), np.round(row[4], 4), np.round(row[5], 4))
def get_past(row): return (np.round(row[1]-row[2], 4), np.round(row[3]-row[6], 4), np.round(row[4]-row[7], 4), np.round(row[5]-row[8], 4))

past_map = defaultdict(list)
vals = df.values

for i in range(len(df)):
    if np.isnan(vals[i, 1]) or np.isnan(vals[i,2]): continue
    past_map[get_past(vals[i])].append(i)

next_node = np.full(len(vals), -1, dtype=int)
for i in range(len(vals)):
    if np.isnan(vals[i,1]): continue
    vec = get_curr(vals[i])
    if vec in past_map:
        cands = past_map[vec]
        if len(cands) == 1:
            next_node[i] = cands[0]

p_arr = vals[:, 1]
ret_1 = np.zeros(len(vals), dtype=np.float32)
ret_2 = np.zeros(len(vals), dtype=np.float32)

for i in range(len(vals)):
    n1 = next_node[i]
    if n1 != -1:
        ret_1[i] = 100.0 * (p_arr[n1] - p_arr[i]) / p_arr[i]
        n2 = next_node[n1]
        if n2 != -1:
            ret_2[i] = 100.0 * (p_arr[n2] - p_arr[i]) / p_arr[i]
        else:
            ret_2[i] = ret_1[i]

# Compute true baseline
math_pred = 0.8044 * ret_1 + 0.1988 * ret_2
del vals, p_arr, ret_1, ret_2; gc.collect()

print("2. Reading 445 Raw Features...")
import pyarrow.parquet as pq
all_cols = pq.ParquetFile('train.parquet').schema.names
feat_cols = [c for c in all_cols if c not in {'ID', 'CV_GROUP', 'TARGET', 'SO3_T'}]

X_train = pd.read_parquet('train.parquet', columns=feat_cols)
X_test = pd.read_parquet('test.parquet', columns=feat_cols)

print("3. Training Deep Residual LightGBM on Base Margin...")

test_preds_ml = np.zeros(len(test_ids), dtype=np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in kf.split(X_train):
    # Set the margin so model EXPLICITLY learns to correct the math
    # init_score is literally base_margin in lightgbm
    dt = lgb.Dataset(X_train.iloc[tr], y_train[tr], init_score=math_pred[:n_tr][tr])
    dv = lgb.Dataset(X_train.iloc[va], y_train[va], init_score=math_pred[:n_tr][va], reference=dt)
    
    params = {
        'objective': 'mse',
        'learning_rate': 0.05,
        'num_leaves': 127,      # Need massive depth to extract the 0.91 logic
        'feature_fraction': 0.7,
        'n_jobs': -1,
        'verbose': -1
    }
    
    bst = lgb.train(params, dt, num_boost_round=600, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # We must ADD the prediction to the math_pred (since lightgbm raw pred is just the residual)
    test_preds_ml += bst.predict(X_test) / 5.0
    
test_preds = math_pred[n_tr:] + test_preds_ml

sub = pd.DataFrame({"ID": test_ids, "TARGET": test_preds})
sub.to_csv("submission_top3_gamble.csv", index=False)

print("4. Done. Good luck!")
