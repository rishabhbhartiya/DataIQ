# Changelog

All notable changes to DataIQ are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2025-05-28

### Added
- **ML Readiness Score** — grades every column A–F across 7 dimensions
  (Completeness, Distribution, Cardinality, Leakage Risk, Outlier Severity,
  Type Fitness, Consistency) with per-column plain-English verdicts
- **Leakage Detective** — 6-category automatic leakage detection
  (Target Correlation, Name Proximity, Derived Feature, Future Data,
  ID/Primary Key, Constant/Zero-Variance)
- **Drift Analyzer** — PSI + KS-test + chi-square train vs production drift
  with per-column reports, missing-rate drift, and schema diff detection
- **Feature Interactions** — pairwise MI / Cramér's V / η² interaction matrix
  with redundancy group detection and feature-vs-target scores
- **Temporal Awareness** — rolling trend charts and concept-drift window
  detection over datetime columns
- **Pipeline Code Generator** — generates complete runnable sklearn Pipeline
  script with auto-selected imputer, scaler, encoder, and model options
- **Narrative Engine** — plain-English summaries for every report section
- **Drift CLI** (`dataiq-drift`) — interactive terminal drift monitor with
  colour-coded PSI table, HTML + JSON output, optional leakage check
- **Drift Report HTML** — standalone drift report with PSI heatmap,
  per-column accordion, missing-rate drift table, schema diff
- **ML Knowledge Base sidebar** — syncs with active nav section,
  resizable by drag, detailed content per section
- **Auto problem-type detection** — infers binary/multiclass/regression
  from target dtype and cardinality
- Universal data loader supporting CSV, Excel, Parquet, JSON, JSONL, XML,
  NumPy arrays, dicts

### Changed
- Library renamed from `mlprofiler2` → `dataiq`
- Main class renamed from `MLProfiler` → `DataIQ`
- HTML template fully redesigned with dark glassmorphism theme
- Chart rendering changed to standalone iframe files (fixes Plotly
  rendering issues in offline HTML)
- Output files now written to `dataiq_output/` directory

### Fixed
- Plotly charts not rendering in offline HTML (iframes fix)
- Smart apostrophe in JS breaking entire script block
- `sys.path` resolution when running `demo.py` from parent directory

---

## [0.2.0] — mlprofiler2 (legacy)

Previous version under the `mlprofiler2` name. Included basic EDA report,
before/after comparison, transformer pipeline, and model training CLI.