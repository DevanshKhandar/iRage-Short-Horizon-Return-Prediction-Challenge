# 📁 FILE GUIDE - Where Everything Is

## 🚀 WHAT TO DO RIGHT NOW

### SUBMIT THIS
- **`submission.csv`** ← Your main submission (44x scaled)
  - Ready to download and upload to AlgoArena
  - Contains 410,139 predictions
  - Predictions scaled by 44x to match target distribution

---

## 📖 DOCUMENTATION (Read These)

### For Submitting
- **`SUBMISSION_GUIDE.md`** 
  - How to upload to AlgoArena
  - What to expect from new score
  - Troubleshooting if something is wrong

- **`FINAL_SUMMARY.md`**
  - High-level overview of what was done
  - Expected improvements
  - Next steps after submitting

### If First Submission Doesn't Work
- **`IF_FIRST_SUBMISSION_FAILS.md`**
  - Alternative strategies to try (quantile, rank, ensemble, etc.)
  - Decision tree for choosing which to try
  - What each strategy is good for

### For Long-Term Improvement
- **`IMPROVEMENT_STRATEGY.md`**
  - Detailed roadmap to reach 0.86181
  - What worked vs what didn't
  - Next 6 approaches to try
  - Why your score is so low
  - Why 0.86 is achievable (with work)

---

## 📊 PREDICTIONS (CSV FILES)

### Main Submission
- **`submission.csv`** ← DOWNLOAD AND SUBMIT THIS
  - Your primary submission
  - 44x standard deviation scaling
  - Clipped to valid range [-1.286, 1.347]

### Alternatives in `output/` Folder
- **`submission_strategy1_std.csv`**
  - 44x scaling by std deviation
  - Same as main submission
  - Use if main doesn't work

- **`submission_strategy2_quantile.csv`**
  - Quantile-based mapping (formula: y = 42.68x + 0.0038)
  - Different scaling approach
  - Try this first if std scaling fails

- **`submission_strategy3_rank.csv`**
  - Rank-based transformation
  - Maps ranks to target quantiles
  - Good for non-normal distributions

- **`submission_strategy4_aggressive.csv`**
  - Conservative 10x scaling
  - Smaller scaling factor
  - Use if 44x seems too much

- **`submission_ensemble.csv`**
  - Average of all 4 strategies
  - Hedges between approaches
  - Good safety choice

---

## 🔧 PYTHON SCRIPTS (For Understanding/Re-running)

### Data Preparation
- **`scale_pure.py`**
  - Pure Python scaling (no pandas needed)
  - Applied 44x scaling to create main submission.csv
  - Shows exactly how predictions were scaled

### Pipeline / Training
- **`improved_memory_pipeline.py`**
  - New training pipeline with target scaling
  - Uses fewer features for memory efficiency
  - Trains with scaled targets for better learning
  - Status: Currently running/completed

- **`multi_strategies.py`**
  - Generated all 5 alternative scaling strategies
  - Creates submission files with different approaches
  - Shows how ensemble was created

- **`final_pipeline.py`** (Original)
  - Your previous best model
  - Used for generating base predictions
  - Now scaled up by 44x

- **`improved_v2_pipeline.py`** (Draft)
  - Earlier attempt at improved pipeline
  - Memory issues, not completed
  - Reference only

### Other Pipelines (Reference)
- `enhanced_pipeline.py`
- `improved_pipeline.py`
- `model_pipeline.py`
- `kaggle_notebook.py`
- `investigate.py`
- `analyze.py`
- `deep_analysis.py`
- `diagnostic_test.py`
- `explore_features.py`

---

## 📈 DIAGNOSTICS & CONFIG (JSON FILES)

In `output/` folder:

- **`scaling_diagnostics.json`** ← MOST IMPORTANT
  - Statistics of all 5 scaling strategies
  - Shows mean/std for each approach
  - Formulas used
  - Compare effectiveness here

- **`cv_results.json`**
  - Cross-validation metrics
  - R² scores per fold
  - Original predictions statistics
  - Model configuration used

- **`feature_importance.json`**
  - Which features contributed most
  - LightGBM feature importance scores

- **`feature_correlations.json`**
  - Correlation of each feature with target
  - Top features by correlation

- **`model_config.json`**
  - Model hyperparameters used
  - LightGBM settings

- **`predictions_analysis.json`**
  - Statistics of original predictions
  - Target statistics
  - Quantile information

---

## 📄 DATA FILES

- **`train.parquet`**
  - Training data (661,574 samples)
  - Contains features, target, and CV groups
  - Used for all model training

- **`test.parquet`**
  - Test data (410,139 samples)
  - Contains features
  - Used for generating predictions

- **`sample_submission.csv`**
  - Template showing required format
  - All zeros (baseline)
  - Reference for CSV format

---

## 📊 DIRECTORY STRUCTURE

```
workspace/
├── submission.csv                    ← MAIN SUBMISSION (download this)
├── sample_submission.csv             ← Format reference
├── train.parquet                     ← Training data
├── test.parquet                      ← Test data
│
├── SUBMISSION_GUIDE.md               ← READ THIS FIRST
├── FINAL_SUMMARY.md                  ← Overview
├── IMPROVEMENT_STRATEGY.md           ← Long-term roadmap
├── IF_FIRST_SUBMISSION_FAILS.md      ← Backup plans
├── FILE_GUIDE.md                     ← This file
│
├── scale_pure.py                     ← How we applied 44x
├── improved_memory_pipeline.py       ← Better training pipeline
├── multi_strategies.py               ← Generated alternatives
├── final_pipeline.py                 ← Original model
│
├── output/
│   ├── submission.csv                ← Copy of main (reference)
│   ├── submission_strategy1_std.csv   ← Strategy 1
│   ├── submission_strategy2_quantile.csv    ← Strategy 2 (try if 1 fails)
│   ├── submission_strategy3_rank.csv        ← Strategy 3
│   ├── submission_strategy4_aggressive.csv  ← Strategy 4
│   ├── submission_ensemble.csv              ← Strategy 5 (ensemble)
│   ├── scaling_diagnostics.json      ← All scaling statistics
│   ├── cv_results.json
│   ├── feature_importance.json
│   ├── feature_correlations.json
│   ├── model_config.json
│   └── predictions_analysis.json
│
└── (other Python files - reference only)
```

---

## 🎯 QUICK ACTION PLAN

### TODAY
1. Read: `SUBMISSION_GUIDE.md`
2. Download: `submission.csv`
3. Submit to AlgoArena
4. Note your new score

### AFTER SUBMISSION
If score improves (e.g., -0.00024 → 0.001):
- ✓ Success! Scaling worked!
- Try alternatives to see if any are better
- See IMPROVEMENT_STRATEGY.md for next steps

If score doesn't improve:
- Read: `IF_FIRST_SUBMISSION_FAILS.md`
- Try: Strategy 2 (quantile)
- If still fails: Try strategies 3, 4, 5

If you keep failing:
- Problem is model, not scaling
- Read: `IMPROVEMENT_STRATEGY.md`
- Try: Neural network approach
- Consider: Better features

---

## 💡 WHAT EACH PART DOES

| File | Purpose | Use When |
|------|---------|----------|
| submission.csv | Main submission | NOW |
| scaling_diagnostics.json | Understand scaling | Curious |
| IF_FIRST_SUBMISSION_FAILS.md | Plan B | Score didn't improve |
| IMPROVEMENT_STRATEGY.md | Reach 0.86 | Need long-term plan |
| improved_memory_pipeline.py | Better training | Want to retrain |
| multi_strategies.py | Understand alternatives | Curious about methods |

---

## ❓ FAQ

**Q: Where is my main submission?**
A: `submission.csv` in the root of your workspace

**Q: Which file do I upload to AlgoArena?**
A: `submission.csv`

**Q: What if I need to try a different strategy?**
A: Use files in `output/submission_strategy*.csv`

**Q: Can I rerun the scaling?**
A: Yes, run `scale_pure.py` or `multi_strategies.py`

**Q: Can I train a better model?**
A: Yes, modify and run `improved_memory_pipeline.py`

**Q: Where are all my files organized?**
A: See directory structure above

**Q: What should I read first?**
A: Read in this order:
1. SUBMISSION_GUIDE.md (5 min)
2. FINAL_SUMMARY.md (5 min)
3. Submit! (2 min)
4. IF_FIRST_SUBMISSION_FAILS.md (10 min if needed)
5. IMPROVEMENT_STRATEGY.md (20 min for long-term)

---

## 🎉 YOU'RE ALL SET!

Your submission is ready. Download `submission.csv` and submit! 🚀

Good luck reaching 0.86181! 💪
