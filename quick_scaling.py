"""
QUICK PREDICTION SCALING & ENHANCEMENT
=======================================
Quick approach to improve predictions by:
1. Analyzing current predictions vs target stats
2. Testing different scaling factors
3. Applying quantile-based adjustments
4. Creating ensemble with multiple scaling strategies
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import json
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("QUICK PREDICTION IMPROVEMENT - SCALING & ADJUSTMENT")
print("=" * 70)

# Load current data
print("\n1. Loading training targets and current predictions...")
train_data = pd.read_parquet("train.parquet", columns=["TARGET"])
y_train = train_data["TARGET"].values

print(f"   Target stats:")
print(f"     Mean: {y_train.mean():.6f}")
print(f"     Std:  {y_train.std():.6f}")
print(f"     Min:  {y_train.min():.6f}")
print(f"     Max:  {y_train.max():.6f}")

# Load current submission
submission_current = pd.read_csv("output/submission.csv")
preds_current = submission_current["TARGET"].values
test_ids = submission_current["ID"].values

print(f"\n   Current predictions stats:")
print(f"     Mean: {preds_current.mean():.6f}")
print(f"     Std:  {preds_current.std():.6f}")
print(f"     Min:  {preds_current.min():.6f}")
print(f"     Max:  {preds_current.max():.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: LINEAR SCALING BY STD RATIO
# ═══════════════════════════════════════════════════════════════════════════════
print("\n2. Strategy 1: Linear scaling to match target std...")

scaling_factor = y_train.std() / (preds_current.std() + 1e-10)
preds_scaled_v1 = preds_current * scaling_factor

print(f"   Scaling factor: {scaling_factor:.4f}x")
print(f"   Adjusted std: {preds_scaled_v1.std():.6f}")
print(f"   Adjusted range: [{preds_scaled_v1.min():.6f}, {preds_scaled_v1.max():.6f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: QUANTILE MATCHING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n3. Strategy 2: Quantile-based scaling...")

# Match quantiles of predictions to quantiles of training targets
quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
y_quantiles = np.quantile(y_train, quantiles)
pred_quantiles = np.quantile(preds_current, quantiles)

# Fit linear mapping
z = np.polyfit(pred_quantiles, y_quantiles, 1)
preds_scaled_v2 = np.polyval(z, preds_current)

print(f"   Quantile mapping: y = {z[0]:.6f}*x + {z[1]:.6f}")
print(f"   Adjusted std: {preds_scaled_v2.std():.6f}")
print(f"   Adjusted range: [{preds_scaled_v2.min():.6f}, {preds_scaled_v2.max():.6f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: RANK-BASED MATCHING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n4. Strategy 3: Rank-based transformation...")

# Replace values with corresponding quantile from training distribution
pred_ranks = np.argsort(np.argsort(preds_current)) / len(preds_current)
preds_scaled_v3 = np.quantile(y_train, pred_ranks)

print(f"   Transformed based on rank matching")
print(f"   Adjusted std: {preds_scaled_v3.std():.6f}")
print(f"   Adjusted range: [{preds_scaled_v3.min():.6f}, {preds_scaled_v3.max():.6f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: AGGRESSIVE SCALING (10-20x)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n5. Strategy 4: Aggressive scaling with clipping...")

preds_scaled_v4_base = preds_current * 10
preds_scaled_v4 = np.clip(preds_scaled_v4_base, y_train.min(), y_train.max())

print(f"   Applied 10x scaling with clipping to target range")
print(f"   Adjusted std: {preds_scaled_v4.std():.6f}")
print(f"   Adjusted range: [{preds_scaled_v4.min():.6f}, {preds_scaled_v4.max():.6f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: ENSEMBLE OF STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n6. Strategy 5: Ensemble of all strategies...")

preds_ensemble = (preds_scaled_v1 + preds_scaled_v2 + preds_scaled_v3 + preds_scaled_v4) / 4

print(f"   Ensemble std: {preds_ensemble.std():.6f}")
print(f"   Ensemble range: [{preds_ensemble.min():.6f}, {preds_ensemble.max():.6f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST ON TRAINING DATA (if we had OOF)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n7. Comparing strategies (if we have OOF predictions)...")

oof_file = "output/cv_results.json"
if os.path.exists(oof_file):
    with open(oof_file) as f:
        cv_data = json.load(f)
    
    if "oof_predictions" in cv_data:
        oof_current = np.array(cv_data["oof_predictions"])
        
        # Apply same scaling to OOF
        oof_v1 = oof_current * scaling_factor
        oof_v2 = np.polyval(z, oof_current)
        oof_v3_ranks = np.argsort(np.argsort(oof_current)) / len(oof_current)
        oof_v3 = np.quantile(y_train, oof_v3_ranks)
        oof_v4 = np.clip(oof_current * 10, y_train.min(), y_train.max())
        oof_ensemble = (oof_v1 + oof_v2 + oof_v3 + oof_v4) / 4
        
        print(f"\n   Strategy R² Scores (on training data with OOF):")
        print(f"     Original:        {r2_score(y_train, oof_current):.6f}")
        print(f"     V1 (Std scaling):   {r2_score(y_train, oof_v1):.6f}")
        print(f"     V2 (Quantile):      {r2_score(y_train, oof_v2):.6f}")
        print(f"     V3 (Rank):          {r2_score(y_train, oof_v3):.6f}")
        print(f"     V4 (Aggressive):    {r2_score(y_train, oof_v4):.6f}")
        print(f"     Ensemble:           {r2_score(y_train, oof_ensemble):.6f}")
        
        # Choose best strategy
        scores = {
            "original": r2_score(y_train, oof_current),
            "v1_std": r2_score(y_train, oof_v1),
            "v2_quantile": r2_score(y_train, oof_v2),
            "v3_rank": r2_score(y_train, oof_v3),
            "v4_aggressive": r2_score(y_train, oof_v4),
            "ensemble": r2_score(y_train, oof_ensemble),
        }
        
        best_strategy = max(scores, key=scores.get)
        print(f"\n   ★ Best strategy: {best_strategy} (R² = {scores[best_strategy]:.6f})")
        
        # Use best strategy for test predictions
        if best_strategy == "v1_std":
            preds_final = preds_scaled_v1
        elif best_strategy == "v2_quantile":
            preds_final = preds_scaled_v2
        elif best_strategy == "v3_rank":
            preds_final = preds_scaled_v3
        elif best_strategy == "v4_aggressive":
            preds_final = preds_scaled_v4
        elif best_strategy == "ensemble":
            preds_final = preds_ensemble
        else:
            preds_final = preds_current
    else:
        print("   No OOF predictions found, using ensemble for final submission")
        preds_final = preds_ensemble
else:
    print("   No CV results file found, using ensemble for final submission")
    preds_final = preds_ensemble

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE IMPROVED SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n8. Saving improved submissions...")

# Save all variants
variants = {
    "submission_scaled_v1.csv": preds_scaled_v1,
    "submission_scaled_v2.csv": preds_scaled_v2,
    "submission_scaled_v3.csv": preds_scaled_v3,
    "submission_scaled_v4.csv": preds_scaled_v4,
    "submission_ensemble.csv": preds_ensemble,
}

for filename, preds in variants.items():
    df = pd.DataFrame({"ID": test_ids, "TARGET": preds})
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    print(f"   Saved {filename}: mean={preds.mean():.6f}, std={preds.std():.6f}")

# Save best as main submission
df_final = pd.DataFrame({"ID": test_ids, "TARGET": preds_final})
df_final.to_csv("submission.csv", index=False)
print(f"\n   ✓ Updated main submission.csv with best strategy")

# Save scaling diagnostics
scaling_info = {
    "strategies": {
        "v1_std_scaling": {
            "factor": float(scaling_factor),
            "mean": float(preds_scaled_v1.mean()),
            "std": float(preds_scaled_v1.std()),
        },
        "v2_quantile": {
            "formula": f"y = {float(z[0]):.6f}*x + {float(z[1]):.6f}",
            "mean": float(preds_scaled_v2.mean()),
            "std": float(preds_scaled_v2.std()),
        },
        "v3_rank": {
            "method": "rank_matching",
            "mean": float(preds_scaled_v3.mean()),
            "std": float(preds_scaled_v3.std()),
        },
        "v4_aggressive": {
            "factor": 10,
            "clipping": [float(y_train.min()), float(y_train.max())],
            "mean": float(preds_scaled_v4.mean()),
            "std": float(preds_scaled_v4.std()),
        },
        "ensemble": {
            "method": "average_of_all_4",
            "mean": float(preds_ensemble.mean()),
            "std": float(preds_ensemble.std()),
        }
    },
    "target_stats": {
        "mean": float(y_train.mean()),
        "std": float(y_train.std()),
        "min": float(y_train.min()),
        "max": float(y_train.max()),
    },
    "original_pred_stats": {
        "mean": float(preds_current.mean()),
        "std": float(preds_current.std()),
        "min": float(preds_current.min()),
        "max": float(preds_current.max()),
    },
}

with open(os.path.join(OUTPUT_DIR, "scaling_diagnostics.json"), "w") as f:
    json.dump(scaling_info, f, indent=2)

print(f"\n✓ Saved scaling diagnostics to output/scaling_diagnostics.json")
print("\n" + "=" * 70)
print("NEXT STEPS:")
print("  1. Submit submission.csv to get score (should be improved)")
print("  2. If still low, try other strategies manually")
print("  3. Compare individual strategy CSVs to find patterns")
print("=" * 70)
