"""
demo.py — DataIQ Full Demo
===========================
Generates all six outputs:
  before.html          — full EDA report (profile)
  compare.html         — before/after comparison
  drift.html           — train vs production drift report
  pipeline.py          — ready-to-run sklearn pipeline script
  employee_cleaned.csv — cleaned dataset
  leakage_report.json  — leakage detective results (printed + saved)

Run
---
  cd dataiq/
  python demo.py

Then open the HTML files in your browser.
For model training:
  python cli/mlprofiler_train.py --data employee_cleaned.csv --target churn
For drift monitoring:
  python cli/dataiq_drift.py --ref employee_cleaned.csv --new employee_prod.csv --target churn
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))          # dataiq/
_ROOT = os.path.dirname(_HERE)                               # mlprofiler/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── All output files go here ──────────────────────────────────────
OUTPUT_DIR = os.path.join(_ROOT, "dataiq_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)   # so relative paths in HTML iframes resolve correctly

import json
import numpy as np
import pandas as pd

from dataiq import DataIQ

# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD DATASET  (same structure as original demo, messiness preserved)
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(42)
n = 600

df = pd.DataFrame({
    "age":        np.random.normal(35, 12, n).clip(18, 80).astype(int),
    "income":     np.random.exponential(55000, n),
    "score":      np.random.normal(72, 15, n).clip(0, 100),
    "tenure":     np.random.poisson(5, n).astype(float),
    "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"], n),
    "city":       np.random.choice(["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad"], n),
    "is_manager": np.random.choice([True, False], n),
    "joined":     pd.date_range("2010-01-01", periods=n, freq="2D"),
    "notes":      ["Employee note " + str(i) + " some longer description here" for i in range(n)],
    "churn":      np.random.choice([0, 0, 0, 1], n),
})

# Inject messiness — same as original demo
for col, pct in [("income", .08), ("score", .15), ("tenure", .45), ("city", .06)]:
    df.loc[np.random.random(n) < pct, col] = np.nan
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)
df.loc[df.sample(10, random_state=2).index, "income"] = 2_500_000

# ── Simulated production data (for drift demo) ──────────────────────────────
np.random.seed(99)
n_prod = 400
df_prod = pd.DataFrame({
    "age":        np.random.normal(40, 14, n_prod).clip(18, 80).astype(int),   # older cohort
    "income":     np.random.exponential(72000, n_prod),                         # higher income
    "score":      np.random.normal(68, 18, n_prod).clip(0, 100),                # slightly lower
    "tenure":     np.random.poisson(4, n_prod).astype(float),
    "department": np.random.choice(["Engineering", "Sales", "Marketing", "Legal"], n_prod),  # new dept
    "city":       np.random.choice(["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Kolkata"], n_prod),
    "is_manager": np.random.choice([True, False], n_prod),
    "joined":     pd.date_range("2016-01-01", periods=n_prod, freq="2D"),
    "notes":      ["Production note " + str(i) for i in range(n_prod)],
    "churn":      np.random.choice([0, 0, 0, 1], n_prod),
})
df_prod.loc[np.random.random(n_prod) < 0.20, "income"] = np.nan   # missing rate jumped

# ══════════════════════════════════════════════════════════════════════════════
# 2. INIT DataIQ
# ══════════════════════════════════════════════════════════════════════════════

sep = "=" * 62

print(sep)
print("  DataIQ — Full Demo")
print(sep)
print(f"  Dataset : Employee Dataset  ({len(df):,} rows × {len(df.columns)} cols)")
print(f"  Target  : churn")
print()

diq = DataIQ(df, name="Employee Dataset", target="churn")
print(repr(diq))

# ══════════════════════════════════════════════════════════════════════════════
# 3. PROFILE  →  before.html
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  STEP 1 — Full EDA Profile")
print(sep)
diq.profile(output="before.html", open_browser=False)

# ══════════════════════════════════════════════════════════════════════════════
# 4. NEW: ML READINESS SCORE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  STEP 2 — ML Readiness Score")
print(sep)
rs = diq.readiness_score()

# Show worst 3 columns in detail
print("\n  Worst columns detail:")
for col_info in rs["top_issues"][:3]:
    print(f"\n    [{col_info['grade']}] {col_info['column']}")
    for dim, verdict in col_info["verdicts"].items():
        print(f"         {dim:<20}: {verdict}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. NEW: LEAKAGE DETECTIVE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  STEP 3 — Leakage Detective")
print(sep)
lr = diq.leakage_report()

# Save leakage report to JSON
with open("leakage_report.json", "w") as f:
    json.dump(lr, f, indent=2, default=str)
print(f"\n  Saved → leakage_report.json")

# ══════════════════════════════════════════════════════════════════════════════
# 6. RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  STEP 4 — Recommendations")
print(sep)
diq.recommend()

# ══════════════════════════════════════════════════════════════════════════════
# 7. APPLY TRANSFORMS  (same as original demo)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  STEP 5 — Apply Transforms")
print(sep)

diq.apply("drop_duplicates")
diq.apply("drop_high_missing")
diq.apply("impute_median")
diq.apply("impute_mode")
diq.apply("cap_outliers")

diq.pipeline_summary()

# ══════════════════════════════════════════════════════════════════════════════
# 8. COMPARE  →  compare.html
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  STEP 6 — Before/After Comparison Report")
print(sep)
diq.compare(output="compare.html", open_browser=False)

# ══════════════════════════════════════════════════════════════════════════════
# 9. NEW: DRIFT REPORT  →  drift.html
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  STEP 7 — Drift Report  (train vs production)")
print(sep)
drift_result = diq.drift(df_prod, output="drift.html", open_browser=False)

print(f"\n  Top drifted columns:")
for col in drift_result["columns"][:5]:
    if col.get("psi") is not None:
        print(f"    {col['column']:<20}  PSI={col['psi']:.4f}  Level={col['drift_level']}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. NEW: GENERATE PIPELINE CODE  →  pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  STEP 8 — Generate sklearn Pipeline Code")
print(sep)
diq.export_pipeline_code("pipeline.py")

# ══════════════════════════════════════════════════════════════════════════════
# 11. EXPORT CLEANED CSV
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  STEP 9 — Export Cleaned CSV")
print(sep)
diq.export_csv("employee_cleaned.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 12. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{sep}")
print("  ✅  DataIQ Demo Complete!")
print(sep)
print()
print(f"  Output folder: dataiq_output/")
print()
print("  Files generated:")
print("    before.html          — full EDA report")
print("    compare.html         — before/after comparison")
print("    drift.html           — train vs production drift")
print("    pipeline.py          — ready-to-run sklearn pipeline")
print("    employee_cleaned.csv — cleaned dataset")
print("    leakage_report.json  — leakage detective findings")
print()
print("  What changed vs original MLProfiler demo:")
print("    + ML Readiness Score  — per-column grades A–F across 7 dimensions")
print("    + Leakage Detective   — 6-category leakage scan saved to JSON")
print("    + Drift Report        — PSI + KS train-vs-production comparison")
print("    + Pipeline Code Gen   — ready-to-run sklearn script")
print()
print("  Next steps:")
print("    python cli/mlprofiler_train.py --data employee_cleaned.csv --target churn")
print("    python cli/dataiq_drift.py --ref employee_cleaned.csv --new employee_prod.csv \\")
print("        --target churn --output drift_output/")
print()