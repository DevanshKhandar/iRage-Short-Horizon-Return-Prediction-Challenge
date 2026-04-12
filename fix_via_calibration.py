"""
PROPER CALIBRATION FOR R2 SCORE
================================
Use quantile-based calibration to match target distribution WITHOUT uniform scaling.
This preserves ranking (critical for R2) while improving distribution fit.
"""

import pandas as pd
import numpy as np

print("="*70)
print("CALIBRATING PREDICTIONS FOR R2 SCORE")
print("="*70)

# Load original submission (baseline score -0.00024)
sub = pd.read_csv('output/submission.csv')
preds_orig = sub['TARGET'].values
ids = sub['ID'].values

# Load training targets to understand distribution
train = pd.read_parquet('train.parquet')
y_true = train['TARGET'].values

print(f"\nOriginal predictions (baseline -0.00024):")
print(f"  Mean: {preds_orig.mean():.8f}")
print(f"  Std:  {preds_orig.std():.8f}")
print(f"  Quantile 50: {np.median(preds_orig):.8f}")

print(f"\nTarget distribution (training data):")
print(f"  Mean: {y_true.mean():.8f}")
print(f"  Std:  {y_true.std():.8f}")
print(f"  Quantile 50: {np.median(y_true):.8f}")

# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION: Linear transformation based on quantiles
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STRATEGY: Quantile-based Linear Calibration")
print("="*70)

# Use multiple quantiles to fit a robust linear transformation
# This ensures predictions maintain ranking while fitting distribution
quantile_points = [0.05, 0.25, 0.50, 0.75, 0.95]

pred_quantiles = []
target_quantiles = []

for q in quantile_points:
    pred_quantiles.append(np.quantile(preds_orig, q))
    target_quantiles.append(np.quantile(y_true, q))

# Fit linear regression: target = slope * pred + intercept
# This gives us the best linear transformation
X = np.column_stack([pred_quantiles, np.ones(len(pred_quantiles))])
y_fit = np.array(target_quantiles)
coeffs, _, _, _ = np.linalg.lstsq(X, y_fit, rcond=None)

slope, intercept = coeffs[0], coeffs[1]

print(f"\nLinear calibration model:")
print(f"  y_calibrated = {slope:.6f} * y_original + {intercept:.8f}")

# Apply calibration to all predictions
preds_calibrated = preds_orig * slope + intercept

print(f"\nCalibrated predictions:")
print(f"  Mean: {preds_calibrated.mean():.8f} (target: {y_true.mean():.8f})")
print(f"  Std:  {preds_calibrated.std():.8f} (target: {y_true.std():.8f})")
print(f"  Quantile 50: {np.median(preds_calibrated):.8f}")

# Verify ranking is preserved
corr_rank = np.corrcoef(preds_orig, preds_calibrated)[0, 1]
print(f"\nRanking preservation:")
print(f"  Correlation orig->calibrated: {corr_rank:.6f} (should be ~1.0)")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CALIBRATED SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

output_df = pd.DataFrame({
    'ID': ids,
    'TARGET': preds_calibrated
})

output_df.to_csv('submission.csv', index=False)

print("\n" + "="*70)
print("SAVED: submission.csv (calibrated)")
print("="*70)
print("\nKey differences from failed 44x scaling:")
print("  ✓ Preserves ranking (R2 depends on this)")
print("  ✓ Matches target distribution (calibrated fit)")
print("  ✓ Minimizes errors (optimal linear transformation)")
print("  ✗ NO uniform scaling (that was the problem!)")

print("\nExpected improvement: Small to moderate (calibration can't create signal)")
print("This provides the best R2 given the model's predictions.")
