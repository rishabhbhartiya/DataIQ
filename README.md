# DataIQ — ML Data Intelligence Suite

> **The ML data library that thinks, not just profiles.**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-dataiq-orange.svg)](https://pypi.org/project/dataiq)
[![Code](https://img.shields.io/badge/code-8k%2B%20lines-lightgrey.svg)]()

Most data profiling tools give you charts and statistics. **DataIQ gives you answers.**

It automatically scores your data for ML readiness, detects leakage before it silently destroys your model, measures feature interactions using information theory, monitors production drift with PSI, and generates a ready-to-run sklearn pipeline — all from a single Python call.

---

## What makes DataIQ different

Every other data profiling tool answers the same question: *"What does my data look like?"*

DataIQ answers a different question: **"Is my data ready to train a model — and what will go wrong if I try?"**

### It scores, not just describes

Instead of showing you a missing value bar chart and leaving the interpretation to you, DataIQ grades every column A–F across 7 ML-specific dimensions. You know immediately which columns are production-ready and which will quietly destroy your model.

### It catches leakage before you train

Data leakage is the most common reason a model with 99% validation accuracy fails completely in production. DataIQ runs 6 independent leakage detectors — name proximity, target correlation, derived features, future data, ID columns, and zero-variance columns — and tells you exactly what to drop and why.

### It measures interactions, not just correlations

Pearson correlation only captures linear relationships. DataIQ uses Mutual Information for numeric pairs, Cramér's V for categorical pairs, and Correlation Ratio η² for mixed pairs. It finds redundant feature groups and tells you which features actually interact with your target — linear or not.

### It watches your data over time

DataIQ detects concept drift within your dataset (mean shift across time quartiles) and measures distribution drift between your training and production data using PSI and KS-tests. You get a colour-coded report telling you which features have drifted and whether to retrain.

### It writes your pipeline for you

After profiling, DataIQ generates a complete, immediately runnable sklearn pipeline script — with the right imputer, scaler, and encoder already chosen for each column based on what it found. Not a template. A script built for your specific data.

### The knowledge base knows what you're looking at

The right sidebar isn't a static glossary. It updates as you navigate — when you open the Skewness section it shows transformation formulas, when you open Leakage it shows the prevention checklist, when you open Class Imbalance it shows SMOTE vs class_weight tradeoffs. The sidebar is also resizable by drag.

---

## Installation

```bash
pip install dataiq
```

**Core dependencies** (auto-installed):
```
pandas >= 1.3    numpy >= 1.21    scipy >= 1.7
plotly >= 5.0    scikit-learn >= 1.0
```

**Optional extras** (for model training CLI):
```bash
pip install dataiq[full]
# installs: shap, optuna, xgboost, lightgbm, pyarrow
```

---

## Quickstart

```python
from dataiq import DataIQ
import pandas as pd

df  = pd.read_csv("your_data.csv")
diq = DataIQ(df, target="churn")

# Full EDA report → before.html
diq.profile("before.html")

# ML Readiness Score — grades every column A–F
diq.readiness_score()

# Leakage Detective — 6-category scan
diq.leakage_report()

# Apply transforms (chainable)
diq.apply("drop_duplicates") \
   .apply("impute_median")   \
   .apply("cap_outliers")

# Before/after comparison → compare.html
diq.compare("compare.html")

# Drift report vs production data → drift.html
df_prod = pd.read_csv("production.csv")
diq.drift(df_prod, "drift.html")

# Export a ready-to-run sklearn pipeline script
diq.export_pipeline_code("pipeline.py")

# Export cleaned data
diq.export_csv("cleaned.csv")
```

---

## Core Features

### 1. ML Readiness Score

Every column is scored 0–100 across **7 dimensions** with a letter grade (A–F):

| Dimension | Weight | What it measures |
|---|---|---|
| Completeness | 2.5× | Missing value rate |
| Leakage Risk | 2.0× | Name proximity + target correlation |
| Outlier Severity | 1.5× | IQR outlier rate |
| Distribution | 1.0× | Skewness + kurtosis combined |
| Cardinality | 1.0× | Encoding complexity risk |
| Type Fitness | 0.8× | How well dtype suits ML |
| Consistency | 0.2× | Constants, mixed types, all-zeros |

```python
report = diq.readiness_score()
print(report["dataset_score"])   # 86.8
print(report["dataset_grade"])   # B
print(report["top_issues"])      # List of worst columns with verdicts
```

### 2. Leakage Detective

Automatically detects **6 categories** of data leakage:

- **Target Correlation** — feature with |r| > 0.85 vs target
- **Name Proximity** — column name shares tokens with target name
- **Derived Feature** — post-hoc derivation of target (e.g. `churn_flag`, `churn_date`)
- **Future Data** — datetime column contains values after a cutoff
- **ID / Primary Key** — high-cardinality identifier memorising training rows
- **Constant / Zero-Variance** — single-value column

```python
report = diq.leakage_report(corr_threshold=0.85, future_cutoff="2024-01-01")
print(report["risk_level"])   # CRITICAL / HIGH / MODERATE / CLEAN
print(report["findings"])     # List with evidence + fix for each finding
```

### 3. Drift Analyzer

Compare train vs production using **PSI + KS-test + chi-square**:

| PSI | Level | Action |
|---|---|---|
| < 0.10 | Negligible | No action needed |
| < 0.20 | Minor | Monitor |
| < 0.25 | Moderate | Plan retraining |
| ≥ 0.25 | **Major** | **Retrain immediately** |

```python
result = diq.drift(df_production, "drift.html")
print(result["verdict"])     # MAJOR DRIFT — 3 column(s) have PSI ≥ 0.25
print(result["avg_psi"])     # 0.67
print(result["n_major"])     # 3
```

Also detects: missing-rate drift, schema changes (appeared/disappeared/dtype-changed columns).

### 4. Feature Interactions

Pairwise interaction strength using the **right method for each column type**:

```python
# Run via profile() or directly:
analysis = diq.analyze()
interactions = analysis["feature_interactions"]
print(interactions["top_pairs"])       # Ranked by score
print(interactions["target_scores"])   # Feature vs target MI scores
print(interactions["redundancy_groups"])  # Clusters of strongly interacting features
```

| Column types | Method | Captures |
|---|---|---|
| Numeric × Numeric | Mutual Information (normalized) | Linear + non-linear |
| Categorical × Categorical | Cramér's V | Chi-square based |
| Numeric × Categorical | Correlation Ratio η² | Variance explained |

### 5. Pipeline Code Generator

Generates a complete, **ready-to-run** sklearn script tailored to your data:

```python
diq.export_pipeline_code("pipeline.py")
```

Automatically decides:
- **Imputer**: KNN (≤5 missing cols) vs Median
- **Scaler**: Robust (outlier-heavy) vs Standard
- **Encoder**: One-Hot (≤15 unique) / Ordinal (16–100) / Frequency (>100)
- **PowerTransformer**: injected for skewed columns
- **SMOTE stub**: included when class imbalance detected
- **Model options**: 3 commented alternatives for classification or regression
- **CV block**: StratifiedKFold for classification, KFold for regression
- **SHAP stub**: ready to uncomment

### 6. Temporal Awareness

For datasets with datetime columns, DataIQ detects **concept drift over time**:

```python
analysis = diq.analyze()
temporal = analysis["temporal_awareness"]
for item in temporal["items"]:
    print(item["datetime_col"], item["has_concept_drift"])
    for finding in item["drift_findings"]:
        print(f"  {finding['feature']}: {finding['drift_pct']:.1f}% mean shift Q1→Q4")
```

---

## HTML Reports

All three reports are self-contained HTML files with:

- **Dark theme** with glassmorphism design
- **Left navigation sidebar** with all sections
- **Right ML Knowledge Base sidebar** — syncs with active section, fully resizable by drag
- **Plotly charts** saved as individual files and embedded via iframes
- **What-If simulator** — interactive readiness score explorer

### Report types

```python
diq.profile("report.html")        # Full EDA report
diq.compare("compare.html")       # Before/after transforms
diq.drift(df_new, "drift.html")   # Train vs production drift
```

---

## Transformations

All transforms are chainable and auto-resolve affected columns from recommendations:

```python
diq.apply("drop_duplicates")
diq.apply("drop_high_missing")    # drops cols with >40% missing
diq.apply("impute_median")        # numeric columns
diq.apply("impute_mode")          # categorical columns
diq.apply("impute_knn")           # KNN imputation
diq.apply("cap_outliers")         # IQR-based Winsorization
diq.apply("remove_outliers")      # drop outlier rows
diq.apply("log_transform")
diq.apply("sqrt_transform")
diq.apply("yeo_johnson")
diq.apply("scale_standard")
diq.apply("scale_minmax")
diq.apply("scale_robust")
diq.apply("encode_onehot")
diq.apply("encode_label")
diq.apply("encode_frequency")
diq.apply("encode_target")

diq.pipeline_summary()   # print what was applied
diq.reset()              # revert to original data
```

---

## CLI Tools

### Model Training

```bash
python -m dataiq.cli.mlprofiler_train \
    --data employee_cleaned.csv \
    --target churn
```

Interactive walkthrough: model selection → preprocessing → cross-validation → SHAP → Optuna hyperparameter tuning → export.

### Drift Monitoring

```bash
python -m dataiq.cli.dataiq_drift \
    --ref train.csv \
    --new production.csv \
    --target churn \
    --cutoff 2024-01-01 \
    --output drift_output/
```

Prints colour-coded PSI table to terminal. Saves `drift_report.html` + `drift_report.json`. Optionally runs leakage check and exports pipeline code.

---

## Data Sources Supported

```python
DataIQ("data.csv")          # CSV
DataIQ("data.xlsx")         # Excel
DataIQ("data.parquet")      # Parquet
DataIQ("data.json")         # JSON / JSON Lines
DataIQ("data.xml")          # XML
DataIQ(df)                  # pandas DataFrame
DataIQ(np_array)            # NumPy array
DataIQ({"col": [...]})      # dict / list of dicts
```

---

## File Structure

```
dataiq/
├── __init__.py
├── core/
│   ├── profiler.py          ← DataIQ orchestrator
│   ├── analyzer.py          ← Full EDA engine (19 sections)
│   ├── scorer.py            ← ML Readiness Score (7 dimensions)
│   ├── leakage.py           ← Leakage Detective (6 categories)
│   ├── drift.py             ← PSI + KS drift analyzer
│   ├── interactions.py      ← MI / Cramér's V / η² pairwise
│   ├── code_gen.py          ← sklearn Pipeline code generator
│   ├── transformer.py       ← 20 preprocessing transforms
│   ├── report_builder.py    ← HTML report renderer
│   └── dataset.py           ← Universal data loader
├── templates/
│   ├── html_template.py     ← EDA + compare report template
│   ├── drift_template.py    ← Drift report template
│   └── narrative.py         ← Plain-English narration engine
├── cli/
│   ├── mlprofiler_train.py  ← Interactive model training CLI
│   └── dataiq_drift.py      ← Drift monitoring CLI
└── demo.py                  ← Full demo (generates all outputs)
```

---

## Full Demo

```bash
git clone https://github.com/yourusername/dataiq.git
cd dataiq
pip install -r requirements.txt
python dataiq/demo.py
```

Generates in `dataiq_output/`:

| File | Description |
|---|---|
| `before.html` | Full EDA report with all sections |
| `compare.html` | Before/after transforms comparison |
| `drift.html` | Train vs production drift report |
| `pipeline.py` | Ready-to-run sklearn pipeline script |
| `employee_cleaned.csv` | Cleaned dataset |
| `leakage_report.json` | Leakage detective findings |

---

## API Reference

```python
DataIQ(data, name=None, target=None)

# Reports
.profile(output, open_browser)          → str (path)
.compare(output, open_browser)          → str (path)
.drift(df_new, output, open_browser)    → dict

# Intelligence
.readiness_score()                      → dict
.leakage_report(corr_threshold, future_cutoff) → dict
.generate_pipeline_code(problem_type)   → str
.export_pipeline_code(path, problem_type) → str

# Analysis
.analyze()                              → dict
.recommend()                            → List[dict]

# Transforms (all chainable, return self)
.apply(action_id, cols, auto)           → DataIQ
.pipeline_summary()                     → None
.reset()                                → DataIQ

# Export
.get_dataframe()                        → pd.DataFrame
.export_csv(path)                       → str
.export_excel(path)                     → str
```

---

## Requirements

```
pandas >= 1.3
numpy >= 1.21
scipy >= 1.7
plotly >= 5.0
scikit-learn >= 1.0
```

Optional:
```
shap          — SHAP feature importance in model training CLI
optuna        — hyperparameter tuning in model training CLI
xgboost       — XGBoost model option
lightgbm      — LightGBM model option
pyarrow       — Parquet file support
statsmodels   — OLS trendlines in bivariate charts
```

---

## License

MIT License — free for personal and commercial use.

---

## Contributing

Pull requests are welcome. For major changes, open an issue first.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push and open a Pull Request

---

*Built with ❤️ — because your data deserves better than a bar chart.*