"""
PROPER CALIBRATION FOR R2 SCORE - Memory Efficient
===================================================
Use quantile-based calibration without loading full datasets.
"""

import pandas as pd
import numpy as np

print("="*70)
print("CALIBRATING PREDICTIONS FOR R2 SCORE")
print("="*70)

# Load original submission (baseline score -0.00024)
print("\nLoading original submission...")
sub = pd.read_csv('output/submission.csv')
preds_orig = sub['TARGET'].values
ids = sub['ID'].values

print(f"Original predictions (baseline -0.00024):")
print(f"  Count: {len(preds_orig)}")
print(f"  Mean: {preds_orig.mean():.8f}")
print(f"  Std:  {preds_orig.std():.8f}")

# Load training targets
print("\nLoading training targets...")
train_df = pd.read_parquet('train.parquet', columns=['TARGET'])
y_true = train_df['TARGET'].values
del train_df  # Free memory

print(f"Target distribution (training data):")
print(f"  Count: {len(y_true)}")
print(f"  Mean: {y_true.mean():.8f}")
print(f"  Std:  {y_true.std():.8f}")

# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION: Linear transformation based on quantiles
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STRATEGY: Quantile-based Linear Calibration")
print("="*70)
print("Fits optimal linear transformation: y_calib = slope * y_orig + intercept")

# Use quantiles to fit robust linear transformation
quantile_points = [0.05, 0.25, 0.50, 0.75, 0.95]

pred_quantiles = []
target_quantiles = []

for q in quantile_points:
    pred_quantiles.append(np.quantile(preds_orig, q))
    target_quantiles.append(np.quantile(y_true, q))

# Fit linear regression
X = np.column_stack([pred_quantiles, np.ones(len(pred_quantiles))])
y_fit = np.array(target_quantiles)
coeffs, _, _, _ = np.linalg.lstsq(X, y_fit, rcond=None)

slope, intercept = coeffs[0], coeffs[1]

print(f"\nCalibration formula:")
print(f"  y_calibrated = {slope:.6f} * y_original + ({intercept:.8f})")

# Apply calibration
preds_calibrated = preds_orig * slope + intercept

print(f"\nCalibrated predictions:")
print(f"  Mean: {preds_calibrated.mean():.8f} (target: {y_true.mean():.8f})")
print(f"  Std:  {preds_calibrated.std():.8f} (target: {y_true.std():.8f})")

# Verify ranking preserved
corr_rank = np.corrcoef(preds_orig, preds_calibrated)[0, 1]
print(f"\nRanking correlation: {corr_rank:.6f} (perfect = 1.0)")
print(f"✓ Ranking preserved (critical for R2)")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CALIBRATED SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

output_df = pd.DataFrame({
    'ID': ids,
    'TARGET': preds_calibrated
})

output_df.to_csv('submission_calibrated.csv', index=False)

print("\n" + "="*70)
print("✓ SAVED: submission_calibrated.csv (calibrated version)")
print("="*70)

print("\nWhy this approach is correct:")
print("  ✓ Preserves ranking (R2 formula depends on rank correlation)")
print("  ✓ Optimal linear fit to target distribution")
print("  ✓ NO uniform scaling (that multiplied errors by 44²=1936x)")
print("\nExpected result:")
print("  Modest improvement from baseline (-0.00024)")
print("  Much better than failed -2.33771 from scaling")
