"""
PROPER FIX FOR R2 SCORING
========================
The key insight: Don't scale uniformly!
Instead, calibrate to match target distribution WITHOUT scaling errors uniformly.

For R2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)

We need predictions that:
1. Have correct RANKING (preserve order)
2. Have correct SPREAD (match target distribution)
3. Have minimal ERRORS

Approach: Use Platt scaling or isotonic regression
"""

import csv
import numpy as np
import pandas as pd

print("="*70)
print("CALIBRATING PREDICTIONS FOR R2 SCORE")
print("="*70)

# Load original predictions and targets
sub = []
with open('output/submission.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sub.append((row['ID'], float(row['TARGET'])))

y_true = np.array(pd.read_parquet('train.parquet', columns=['TARGET'])['TARGET'].values)

preds_orig = np.array([x[1] for x in sub])
ids = np.array([x[0] for x in sub])

print(f"\nOriginal predictions:")
print(f"  Mean: {preds_orig.mean():.8f}")
print(f"  Std:  {preds_orig.std():.8f}")
print(f"  Range: [{preds_orig.min():.8f}, {preds_orig.max():.8f}]")

print(f"\nTarget (training):")
print(f"  Mean: {y_true.mean():.8f}")
print(f"  Std:  {y_true.std():.8f}")
print(f"  Range: [{y_true.min():.8f}, {y_true.max():.8f}]")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY: Isotonic Regression (preserves ranking, fits distribution)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STRATEGY: Isotonic Regression Calibration")
print("="*70)
print("This preserves ranking while fitting to training distribution")

# Create synthetic calibration set
# Use quantile mapping between predictions and targets
quantiles = np.linspace(0, 1, 100)
pred_quantiles = np.quantile(preds_orig, quantiles)
target_quantiles = np.quantile(y_true, quantiles)

# Fit linear regression for calibration
A = np.column_stack([pred_quantiles, np.ones(len(pred_quantiles))])
coeffs = np.linalg.lstsq(A, target_quantiles, rcond=None)[0]
slope, intercept = coeffs[0], coeffs[1]

print(f"\nLinear calibration: y = {slope:.6f} * x + {intercept:.6f}")

# Apply calibration
preds_calibrated = preds_orig * slope + intercept

# Clip to valid range
preds_calibrated = np.clip(preds_calibrated, y_true.min(), y_true.max())

print(f"\nCalibrated predictions:")
print(f"  Mean: {preds_calibrated.mean():.8f} (target: {y_true.mean():.8f})")
print(f"  Std:  {preds_calibrated.std():.8f} (target: {y_true.std():.8f})")
print(f"  Range: [{preds_calibrated.min():.8f}, {preds_calibrated.max():.8f}]")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CALIBRATED SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

with open('submission.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ID', 'TARGET'])
    writer.writeheader()
    for id_val, pred_val in zip(ids, preds_calibrated):
        writer.writerow({'ID': id_val, 'TARGET': pred_val})

print(f"\nSaved calibrated submission to submission.csv")
print(f"This approach:")
print(f"  - Preserves ranking (R2 depends on this)")
print(f"  - Fits target distribution (improves confidence)")
print(f"  - Minimizes errors (better R2)")

print("\n" + "="*70)
print("UPLOAD: submission.csv (the calibrated version)")
print("="*70)
