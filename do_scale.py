import pandas as pd
import numpy as np

sub = pd.read_csv('output/submission.csv')
y = pd.read_parquet('train.parquet', columns=['TARGET'])['TARGET'].values
scale = y.std() / (sub['TARGET'].std() + 1e-10)
sub['TARGET'] = np.clip(sub['TARGET'] * scale, y.min(), y.max())
sub.to_csv('submission.csv', index=False)
print(f'✓ Scaled by {scale:.2f}x')
print(f'  New std: {sub["TARGET"].std():.6f} (target: {y.std():.6f})')
print(f'  Range: [{sub["TARGET"].min():.6f}, {sub["TARGET"].max():.6f}]')
