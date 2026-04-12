"""
VERIFICATION SCRIPT - Confirm all deliverables are ready
"""
import os
import csv

print("="*70)
print("VERIFICATION REPORT - AlgoArena Challenge Improvements")
print("="*70)

checks = {
    "Main Submission": False,
    "Alternative Strategies": False,
    "Documentation": False,
    "Diagnostics": False,
    "Code Scripts": False,
}

print("\n1. CHECKING MAIN SUBMISSION...")
if os.path.exists("submission.csv"):
    with open("submission.csv", 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"   [OK] submission.csv exists")
    print(f"   [OK] Rows: {len(rows)} (expected: 410139)")
    if len(rows) == 410139:
        print(f"   [OK] Correct number of rows!")
    # Check values
    sample_vals = [float(row['TARGET']) for row in rows[:5]]
    print(f"   [OK] Sample values: {sample_vals}")
    checks["Main Submission"] = len(rows) == 410139
else:
    print("   [FAIL] submission.csv NOT FOUND")

print("\n2. CHECKING ALTERNATIVE STRATEGIES...")
strategies = [
    "output/submission_strategy1_std.csv",
    "output/submission_strategy2_quantile.csv",
    "output/submission_strategy3_rank.csv",
    "output/submission_strategy4_aggressive.csv",
    "output/submission_ensemble.csv",
]
found = 0
for strat in strategies:
    if os.path.exists(strat):
        print(f"   ✓ {strat}")
        found += 1
    else:
        print(f"   ✗ {strat} MISSING")

checks["Alternative Strategies"] = found >= 4
print(f"   Summary: {found}/5 strategies found")

print("\n3. CHECKING DOCUMENTATION...")
docs = [
    "SUBMISSION_GUIDE.md",
    "FINAL_SUMMARY.md",
    "IMPROVEMENT_STRATEGY.md",
    "IF_FIRST_SUBMISSION_FAILS.md",
    "FILE_GUIDE.md",
]
found_docs = 0
for doc in docs:
    if os.path.exists(doc):
        print(f"   ✓ {doc}")
        found_docs += 1
    else:
        print(f"   ✗ {doc} MISSING")

checks["Documentation"] = found_docs == 5
print(f"   Summary: {found_docs}/5 documentation files found")

print("\n4. CHECKING DIAGNOSTICS...")
diags = [
    "output/scaling_diagnostics.json",
    "output/cv_results.json",
    "output/model_config.json",
]
found_diags = 0
for diag in diags:
    if os.path.exists(diag):
        print(f"   ✓ {diag}")
        found_diags += 1

checks["Diagnostics"] = found_diags >= 1
print(f"   Summary: {found_diags} diagnostics found")

print("\n5. CHECKING PYTHON SCRIPTS...")
scripts = [
    "scale_pure.py",
    "improved_memory_pipeline.py",
    "multi_strategies.py",
]
found_scripts = 0
for script in scripts:
    if os.path.exists(script):
        print(f"   ✓ {script}")
        found_scripts += 1

checks["Code Scripts"] = found_scripts >= 2
print(f"   Summary: {found_scripts}/3 scripts found")

print("\n" + "="*70)
print("FINAL STATUS")
print("="*70)

all_passed = all(checks.values())

for check, status in checks.items():
    symbol = "✓" if status else "✗"
    print(f"{symbol} {check}")

print("\n" + "="*70)
if all_passed:
    print("✅ ALL CHECKS PASSED - READY TO SUBMIT!")
    print("\nNext steps:")
    print("1. Download submission.csv")
    print("2. Submit to AlgoArena competition")
    print("3. Check your new leaderboard score")
    print("4. Read IF_FIRST_SUBMISSION_FAILS.md if needed")
else:
    print("⚠️  SOME CHECKS FAILED - Please review above")

print("="*70)
