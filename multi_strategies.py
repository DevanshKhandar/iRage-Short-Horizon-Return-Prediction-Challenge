"""Multiple scaling strategies for predictions"""
import csv
import json

print("="*70)
print("ALTERNATIVE SCALING STRATEGIES")
print("="*70)

# Read current submissions CSV
rows = []
with open('output/submission.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

preds_original = [float(row['TARGET']) for row in rows]
ids = [row['ID'] for row in rows]

# Load target stats
try:
    with open('output/predictions_analysis.json', 'r') as f:
        stats = json.load(f)
        target_mean = stats.get('target_mean', -0.000036)
        target_std = stats.get('target_std', 0.036660)
        target_min = stats.get('target_min', -1.285424)
        target_max = stats.get('target_max', 1.346982)
except:
    target_mean = -0.000036
    target_std = 0.036660
    target_min = -1.285424
    target_max = 1.346982

print(f"\nTarget statistics:")
print(f"  Mean: {target_mean:.8f}")
print(f"  Std:  {target_std:.8f}")
print(f"  Range: [{target_min:.6f}, {target_max:.6f}]")

n = len(preds_original)
pred_mean = sum(preds_original) / n
pred_var = sum((p - pred_mean) ** 2 for p in preds_original) / n
pred_std = pred_var ** 0.5

print(f"\nOriginal predictions statistics:")
print(f"  Mean: {pred_mean:.8f}")
print(f"  Std:  {pred_std:.8f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: STD SCALING (already applied)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 1: Standard Deviation Scaling")
print("="*70)

scale1 = target_std / (pred_std + 1e-10)
preds_v1 = [min(target_max, max(target_min, p * scale1)) for p in preds_original]

new_mean_v1 = sum(preds_v1) / n
new_std_v1 = (sum((p - new_mean_v1) ** 2 for p in preds_v1) / n) ** 0.5

print(f"Scaling factor: {scale1:.4f}x")
print(f"Result - Mean: {new_mean_v1:.8f}, Std: {new_std_v1:.8f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: QUANTILE MATCHING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 2: Quantile Matching")
print("="*70)

# Simulated target quantiles (from actual data)
quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
# These are approximate quantiles from analysis
target_quantile_vals = [-0.101, -0.052, -0.016, -0.000068, 0.016, 0.052, 0.101]
pred_sorted = sorted(preds_original)

# Map prediction quantiles to target quantiles
pred_quantile_indices = [int(i * (n-1)) for i in quantiles]
pred_quantile_vals = [pred_sorted[i] if i < len(pred_sorted) else pred_sorted[-1] for i in pred_quantile_indices]

# Fit linear mapping
sum_x = sum(pred_quantile_vals)
sum_y = sum(target_quantile_vals)
sum_xy = sum(x*y for x, y in zip(pred_quantile_vals, target_quantile_vals))
sum_xx = sum(x*x for x in pred_quantile_vals)

m = (len(quantiles) * sum_xy - sum_x * sum_y) / (len(quantiles) * sum_xx - sum_x * sum_x + 1e-10)
b = (sum_y - m * sum_x) / len(quantiles)

print(f"Linear mapping: y = {m:.6f} * x + {b:.6f}")

preds_v2 = [min(target_max, max(target_min, m * p + b)) for p in preds_original]

new_mean_v2 = sum(preds_v2) / n
new_std_v2 = (sum((p - new_mean_v2) ** 2 for p in preds_v2) / n) ** 0.5

print(f"Result - Mean: {new_mean_v2:.8f}, Std: {new_std_v2:.8f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: RANK-BASED TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 3: Rank-Based Transformation")
print("="*70)

# For each prediction, get its rank and map to corresponding target value
# Create rank mapping
pred_indices = list(range(n))
pred_indices.sort(key=lambda i: preds_original[i])

# Generate synthetic target samples at same quantiles
preds_v3 = []
for i in range(n):
    rank = pred_indices.index(i) / n  # 0 to 1
    
    # Map rank to target using quantiles
    if rank <= 0.01:
        target_val = target_min + rank * (target_quantile_vals[0] - target_min) / 0.01
    elif rank >= 0.99:
        target_val = target_quantile_vals[-1] + (rank - 0.99) * (target_max - target_quantile_vals[-1]) / 0.01
    else:
        # Linear interpolation between quantiles
        for j in range(len(quantiles) - 1):
            if quantiles[j] <= rank <= quantiles[j+1]:
                frac = (rank - quantiles[j]) / (quantiles[j+1] - quantiles[j])
                target_val = target_quantile_vals[j] + frac * (target_quantile_vals[j+1] - target_quantile_vals[j])
                break
    
    preds_v3.append(target_val)

new_mean_v3 = sum(preds_v3) / n
new_std_v3 = (sum((p - new_mean_v3) ** 2 for p in preds_v3) / n) ** 0.5

print(f"Rank-based transformation applied")
print(f"Result - Mean: {new_mean_v3:.8f}, Std: {new_std_v3:.8f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: AGGRESSIVE SCALING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 4: Aggressive 10x Scaling")
print("="*70)

preds_v4 = [min(target_max, max(target_min, p * 10)) for p in preds_original]

new_mean_v4 = sum(preds_v4) / n
new_std_v4 = (sum((p - new_mean_v4) ** 2 for p in preds_v4) / n) ** 0.5

print(f"Applied 10x scaling with clipping")
print(f"Result - Mean: {new_mean_v4:.8f}, Std: {new_std_v4:.8f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 5: Ensemble Average")
print("="*70)

preds_ensemble = [(preds_v1[i] + preds_v2[i] + preds_v3[i] + preds_v4[i]) / 4 for i in range(n)]

new_mean_ens = sum(preds_ensemble) / n
new_std_ens = (sum((p - new_mean_ens) ** 2 for p in preds_ensemble) / n) ** 0.5

print(f"Ensemble of all 4 strategies")
print(f"Result - Mean: {new_mean_ens:.8f}, Std: {new_std_ens:.8f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE ALL VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SAVING VARIANTS")
print("="*70)

variants = {
    'submission_strategy1_std.csv': preds_v1,
    'submission_strategy2_quantile.csv': preds_v2,
    'submission_strategy3_rank.csv': preds_v3,
    'submission_strategy4_aggressive.csv': preds_v4,
    'submission_ensemble.csv': preds_ensemble,
}

for filename, preds in variants.items():
    with open(f'output/{filename}', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'TARGET'])
        writer.writeheader()
        for id_val, pred_val in zip(ids, preds):
            writer.writerow({'ID': id_val, 'TARGET': pred_val})
    print(f"  ✓ {filename}")

print(f"\nUsing ensemble as main submission...")
with open('submission.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ID', 'TARGET'])
    writer.writeheader()
    for id_val, pred_val in zip(ids, preds_ensemble):
        writer.writerow({'ID': id_val, 'TARGET': pred_val})

print(f"✓ Updated submission.csv with ensemble predictions")
print("\nSummary of all strategies:")
print(f"  V1 (Std):      Mean={new_mean_v1:+.8f}, Std={new_std_v1:.8f}")
print(f"  V2 (Quantile): Mean={new_mean_v2:+.8f}, Std={new_std_v2:.8f}")
print(f"  V3 (Rank):     Mean={new_mean_v3:+.8f}, Std={new_std_v3:.8f}")
print(f"  V4 (Aggressive):Mean={new_mean_v4:+.8f}, Std={new_std_v4:.8f}")
print(f"  Ensemble:      Mean={new_mean_ens:+.8f}, Std={new_std_ens:.8f}")
