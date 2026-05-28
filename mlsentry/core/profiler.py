"""
profiler.py — MLRadar
======================
Main orchestrator. Renamed from MLProfiler → MLRadar.

Original methods (preserved exactly, just class renamed):
    profile()          → full EDA report
    compare()          → before/after comparison report
    analyze()          → run analysis, cache result
    recommend()        → print recommendations
    apply()            → apply a transformation (chainable)
    pipeline_summary() → print transform log
    get_dataframe()    → return current DataFrame
    export_csv()       → save to CSV
    export_excel()     → save to Excel
    reset()            → revert to original data

New methods added:
    drift()                  → PSI + KS drift report vs a second DataFrame
    readiness_score()        → ML Readiness Score per column + dataset
    leakage_report()         → Leakage Detective (6 categories)
    generate_pipeline_code() → returns sklearn Pipeline script as string
    export_pipeline_code()   → saves that script to a .py file
"""
from __future__ import annotations

import webbrowser
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from .dataset      import Dataset
from .analyzer     import AdvancedAnalyzer
from .transformer  import Transformer
from .report_builder import ReportBuilder
from .scorer       import ReadinessScorer
from .leakage      import LeakageDetector
from .drift        import DriftAnalyzer
from .code_gen     import PipelineCodeGenerator

warnings.filterwarnings("ignore")


class MLRadar:
    """
    MLRadar — Advanced ML Data Intelligence Suite.

    Parameters
    ----------
    data   : pd.DataFrame | file path (csv / excel / parquet / json / xml)
    name   : display name for the dataset (auto-inferred if omitted)
    target : name of the target column (optional but recommended)

    Example
    -------
    >>> diq = MLRadar(df, target="churn")
    >>> diq.profile("report.html")
    >>> diq.apply("drop_duplicates").apply("impute_median").apply("cap_outliers")
    >>> diq.compare("compare.html")
    >>> diq.drift(df_test, "drift.html")
    >>> print(diq.readiness_score())
    >>> print(diq.leakage_report())
    >>> diq.export_pipeline_code("pipeline.py")
    """

    def __init__(self, data, name: Optional[str] = None,
                 target: Optional[str] = None, **kw):
        self.dataset  = Dataset(data, name=name, target=target, **kw)
        self.target   = target
        self._current = self.dataset.df.copy()
        self._history: List[Dict[str, Any]] = []
        self._analysis_before: Optional[Dict[str, Any]] = None
        self._analysis_after:  Optional[Dict[str, Any]] = None

    # ══════════════════════════════════════════════════════════════════
    # ORIGINAL METHODS  (preserved from MLProfiler, class name only changed)
    # ══════════════════════════════════════════════════════════════════

    def profile(self, output: str = "MLRadar_report.html",
                open_browser: bool = True) -> str:
        """Generate full EDA report for the current dataset."""
        from pathlib import Path
        from .analyzer import set_charts_dir
        charts_dir = str(Path(output).stem) + "_charts"
        set_charts_dir(charts_dir)
        print(f"🔬 Analyzing '{self.dataset.name}' "
              f"({self._current.shape[0]:,}×{self._current.shape[1]})...")
        a = AdvancedAnalyzer(self._current, self.target, label="current")
        self._analysis_before = a.analyze()
        path = ReportBuilder(
            self._analysis_before,
            dataset_name=self.dataset.name,
        ).render(output)
        print(f"✅ Report → {path}")
        if open_browser:
            try: webbrowser.open(f"file://{path}")
            except Exception: pass
        return path

    def compare(self, output: str = "MLRadar_compare.html",
                open_browser: bool = True) -> str:
        """
        Generate a before/after comparison report.
        Call profile() (or apply transforms) first, then compare().
        """
        from pathlib import Path
        from .analyzer import set_charts_dir

        if self._analysis_before is None:
            print("⚠  Running initial analysis first...")
            set_charts_dir(str(Path(output).stem) + "_before_charts")
            a = AdvancedAnalyzer(self.dataset.df, self.target, label="before")
            self._analysis_before = a.analyze()

        print(f"🔬 Analyzing transformed dataset "
              f"({self._current.shape[0]:,}×{self._current.shape[1]})...")
        set_charts_dir(str(Path(output).stem) + "_after_charts")
        a2 = AdvancedAnalyzer(self._current, self.target, label="after")
        self._analysis_after = a2.analyze()

        path = ReportBuilder(
            self._analysis_before,
            self._analysis_after,
            dataset_name=self.dataset.name,
        ).render(output)
        print(f"✅ Comparison report → {path}")
        if open_browser:
            try: webbrowser.open(f"file://{path}")
            except Exception: pass
        return path

    def analyze(self) -> Dict[str, Any]:
        """Run full analysis on current data and cache the result."""
        a = AdvancedAnalyzer(self._current, self.target, label="current")
        self._analysis_before = a.analyze()
        return self._analysis_before

    def recommend(self) -> List[Dict[str, Any]]:
        """Print recommendations to stdout and return them as a list."""
        if not self._analysis_before:
            self.analyze()
        recs = self._analysis_before.get("recommendations", [])
        print(f"\n📋  {len(recs)} Recommendations:\n")
        for i, r in enumerate(recs):
            print(f"  [{i+1}] [{r['severity'].upper()}] {r['title']}")
            print(f"       {r['description']}")
            for a in r.get("actions", []):
                print(f"       → '{a['action_id']}' — {a['label']}")
            print()
        return recs

    def apply(self, action_id: str,
              cols: Optional[List[str]] = None,
              auto: bool = True) -> "MLRadar":
        """
        Apply a preprocessing transformation (chainable).

        Parameters
        ----------
        action_id : str           — transformation name (see Transformer.ACTIONS)
        cols      : List[str]     — columns to apply to; None = auto from recs
        auto      : bool          — resolve cols from cached recommendations
        """
        if auto and cols is None and self._analysis_before:
            for r in self._analysis_before.get("recommendations", []):
                for a in r.get("actions", []):
                    if a["action_id"] == action_id:
                        cols = r.get("affected_columns")
                        break

        t      = Transformer(self._current)
        result = t.apply(action_id, cols, self.target)
        rb, ra = len(self._current), len(result)
        cb, ca = len(self._current.columns), len(result.columns)

        self._history.append({
            "action":       action_id,
            "cols":         cols or [],
            "rows_removed": rb - ra,
            "cols_removed": cb - ca,
        })
        self._current         = result
        self._analysis_before = None    # invalidate cached analysis
        print(f"✅ {action_id}: rows {rb:,}→{ra:,} (-{rb-ra}), "
              f"cols {cb}→{ca} (-{cb-ca})")
        return self

    def pipeline_summary(self) -> None:
        """Print the applied transformation pipeline to stdout."""
        if not self._history:
            print("No transforms applied.")
            return
        print("\n📜 Transformation Pipeline:")
        for i, h in enumerate(self._history):
            print(f"  {i+1}. {h['action']}  "
                  f"cols={h['cols'][:3]}  "
                  f"rows_removed={h['rows_removed']}")
        print(f"\n  Original: {self.dataset.df.shape}  "
              f"→  Current: {self._current.shape}")

    def get_dataframe(self) -> pd.DataFrame:
        """Return a copy of the current (possibly transformed) DataFrame."""
        return self._current.copy()

    def export_csv(self, path: str = "cleaned.csv") -> str:
        """Save the current DataFrame to a CSV file."""
        self._current.to_csv(path, index=False)
        print(f"💾 {path}")
        return path

    def export_excel(self, path: str = "cleaned.xlsx") -> str:
        """Save the current DataFrame to an Excel file."""
        self._current.to_excel(path, index=False)
        print(f"💾 {path}")
        return path

    def reset(self) -> "MLRadar":
        """Revert all transforms and reset to the original data."""
        self._current         = self.dataset.df.copy()
        self._history         = []
        self._analysis_before = None
        self._analysis_after  = None
        print("🔄 Reset to original.")
        return self

    def __repr__(self) -> str:
        return (f"MLRadar('{self.dataset.name}', "
                f"shape={self._current.shape}, "
                f"target={self.target!r}, "
                f"transforms={len(self._history)})")

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 1 — Drift Report
    # ══════════════════════════════════════════════════════════════════

    def drift(self, df_new: pd.DataFrame,
              output: str = "MLRadar_drift.html",
              open_browser: bool = True) -> Dict[str, Any]:
        """
        Compare the current dataset (reference/train) against df_new
        (test / production) using PSI, KS-test, and chi-square.

        Parameters
        ----------
        df_new       : pd.DataFrame — new dataset to compare against
        output       : str          — output HTML file path
        open_browser : bool         — open the report in a browser

        Returns
        -------
        dict — full drift report from DriftAnalyzer.analyze()
        """
        print(f"📡 Running drift analysis  "
              f"(ref: {len(self._current):,} rows  →  "
              f"new: {len(df_new):,} rows)...")

        da     = DriftAnalyzer(self._current, df_new, target=self.target)
        result = da.analyze()

        print(f"   {result['verdict']}")
        print(f"   Avg PSI: {result['avg_psi']}  |  "
              f"Major: {result['n_major']}  "
              f"Moderate: {result['n_moderate']}  "
              f"Minor: {result['n_minor']}  "
              f"Negligible: {result['n_negligible']}")
        if result["missing_drift"]:
            print(f"   Missing-rate drift in "
                  f"{len(result['missing_drift'])} column(s)")
        if result["schema"]["has_changes"]:
            sc = result["schema"]
            print(f"   Schema changes — "
                  f"appeared: {sc['appeared']}  "
                  f"disappeared: {sc['disappeared']}")

        path = ReportBuilder(
            analysis_before=self._analysis_before or {},
            drift_result=result,
            dataset_name=self.dataset.name,
        ).render_drift(output)

        print(f"✅ Drift report → {path}")
        if open_browser:
            try: webbrowser.open(f"file://{path}")
            except Exception: pass
        return result

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 2 — ML Readiness Score
    # ══════════════════════════════════════════════════════════════════

    def readiness_score(self) -> Dict[str, Any]:
        """
        Score every column and the full dataset on ML readiness (0–100, A–F).

        Seven dimensions scored per column:
            Completeness, Distribution, Cardinality, Leakage Risk,
            Outlier Severity, Type Fitness, Consistency

        Returns
        -------
        dict — full report from ReadinessScorer.score()
            keys: dataset_score, dataset_grade, dimension_avgs,
                  grade_dist, columns, top_issues, strong_cols
        """
        scorer = ReadinessScorer(self._current, self.target)
        result = scorer.score()

        score = result["dataset_score"]
        grade = result["dataset_grade"]
        label = result["dataset_label"]
        dist  = result["grade_dist"]

        print(f"\n📊 ML Readiness Score: {score}/100  "
              f"Grade {grade}  —  {label}")
        print(f"   Columns: "
              f"A={dist['A']}  B={dist['B']}  C={dist['C']}  "
              f"D={dist['D']}  F={dist['F']}")

        if result["top_issues"]:
            names = ", ".join(
                f"'{c['column']}' ({c['grade']})"
                for c in result["top_issues"][:5]
            )
            print(f"   ⚠  Lowest scoring: {names}")

        if result["strong_cols"]:
            names = ", ".join(
                f"'{c['column']}'"
                for c in result["strong_cols"][:5]
            )
            print(f"   ✓  Strongest cols: {names}")

        return result

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 3 — Leakage Report
    # ══════════════════════════════════════════════════════════════════

    def leakage_report(self, corr_threshold: float = 0.85,
                       future_cutoff: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the Leakage Detective across six leakage categories:
            Target Correlation, Name Proximity, Derived Feature,
            Future Data, ID / Primary Key, Constant / Zero-Variance

        Parameters
        ----------
        corr_threshold : float — Pearson |r| threshold for correlation leakage
        future_cutoff  : str   — ISO date string for future-data check

        Returns
        -------
        dict — full report from LeakageDetector.detect()
            keys: findings, n_critical, n_high, n_medium, n_low,
                  n_total, safe_cols, risk_level, verdict
        """
        ld     = LeakageDetector(
            self._current,
            target=self.target,
            corr_threshold=corr_threshold,
            future_cutoff=future_cutoff,
        )
        result = ld.detect()

        print(f"\n🔍 Leakage Detective: {result['verdict']}")
        print(f"   {result['n_total']} finding(s) — "
              f"critical={result['n_critical']}  "
              f"high={result['n_high']}  "
              f"medium={result['n_medium']}  "
              f"low={result['n_low']}")

        for f in result["findings"]:
            sev = f["severity"].upper()
            print(f"\n   {f['icon']} [{sev}]  {f['column']}")
            print(f"      Category : {f['category']}")
            print(f"      Evidence : {f['evidence']}")
            print(f"      Fix      : {f['fix']}")

        if result["safe_cols"]:
            print(f"\n   ✓ Safe columns: {result['safe_cols']}")

        return result

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD 4 — Pipeline Code Generator
    # ══════════════════════════════════════════════════════════════════

    def generate_pipeline_code(self,
                                problem_type: Optional[str] = None) -> str:
        """
        Generate a complete, runnable sklearn Pipeline script tailored
        to this dataset's column types, missing rates, outliers, and
        class balance.

        Parameters
        ----------
        problem_type : "classification" | "regression" | None (auto-detect)

        Returns
        -------
        str — full Python script content
        """
        if self._analysis_before is None:
            self.analyze()

        gen  = PipelineCodeGenerator(
            analysis=self._analysis_before,
            target=self.target,
            problem_type=problem_type,
        )
        code = gen.generate()
        print(f"\n🧬 Pipeline code generated  "
              f"({len(code.splitlines())} lines)")
        print(f"   Problem type : {gen.problem_type}")
        print(f"   Numeric cols : {gen.num_cols}")
        print(f"   OHE cols     : {gen.cat_ohe}")
        print(f"   Ordinal cols : {gen.cat_ord}")
        print(f"   Freq cols    : {gen.cat_freq}")
        print(f"   Dropped      : {gen.drop_cols}")
        print(f"   Scaler       : {'Robust' if gen.use_robust else 'Standard'}")
        print(f"   Imputer      : {'KNN' if gen.use_knn else 'Median/Mode'}")
        if gen.is_imbalanced:
            print(f"   ⚠  Imbalanced target "
                  f"(ratio ≈ {gen.imbalance_ratio:.1f}×) — "
                  f"class_weight set + SMOTE stub included")
        return code

    def export_pipeline_code(self, path: str = "pipeline.py",
                              problem_type: Optional[str] = None) -> str:
        """
        Generate and save the sklearn Pipeline script to a .py file.

        Parameters
        ----------
        path         : str — output file path
        problem_type : str — "classification" | "regression" | None

        Returns
        -------
        str — resolved absolute file path
        """
        code     = self.generate_pipeline_code(problem_type=problem_type)
        out_path = Path(path)
        out_path.write_text(code, encoding="utf-8")
        print(f"💾 Pipeline code → {out_path.resolve()}")
        return str(out_path.resolve())