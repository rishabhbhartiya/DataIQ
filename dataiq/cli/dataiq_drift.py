#!/usr/bin/env python3
"""
dataiq_drift.py — Interactive Drift Monitor CLI
=================================================
Compare a reference dataset (train) against a new dataset (test / production)
and produce a PSI + KS-test drift report.

Usage
-----
  python dataiq_drift.py --ref train.csv --new production.csv
  python dataiq_drift.py --ref train.csv --new production.csv --target churn
  python dataiq_drift.py --ref train.csv --new production.csv \\
      --target churn --cutoff 2024-01-01 --output drift_output/ --no-browser

Walkthrough
-----------
  1. Load reference + new datasets
  2. Confirm or choose target column
  3. Set optional future-data cutoff date
  4. Run DriftAnalyzer (PSI + KS + chi-square + schema diff + missing drift)
  5. Print colour-coded per-column results to terminal
  6. Save HTML drift report
  7. Save JSON result (machine-readable)
  8. Ask whether to run leakage check on reference data
  9. Ask whether to export pipeline code for the reference data
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── colour helpers (same pattern as mlprofiler_train.py) ──────────────────────
COLORS = {
    "cyan":    "\033[96m", "green":  "\033[92m", "yellow": "\033[93m",
    "red":     "\033[91m", "bold":   "\033[1m",  "dim":    "\033[2m",
    "blue":    "\033[94m", "magenta":"\033[95m",  "reset":  "\033[0m",
    "orange":  "\033[33m",
}


def c(text, *clrs) -> str:
    if not sys.stdout.isatty():
        return str(text)
    codes = "".join(COLORS.get(x, "") for x in clrs)
    return f"{codes}{text}{COLORS['reset']}"


def header(msg: str) -> None:
    w = 66
    print()
    print(c("┌" + "─" * (w - 2) + "┐", "cyan"))
    print(c("│ " + msg.center(w - 4) + " │", "cyan", "bold"))
    print(c("└" + "─" * (w - 2) + "┘", "cyan"))
    print()


def section(msg: str) -> None:
    print()
    print(c(f"  ━━━  {msg}  ━━━", "blue", "bold"))
    print()


def ask(prompt: str, default=None) -> str:
    hint = f" [{default}]" if default is not None else ""
    raw  = input(c(f"  ❯ {prompt}{hint}: ", "yellow")).strip()
    return raw if raw else (str(default) if default is not None else "")


def confirm(msg: str, default: bool = True) -> bool:
    hint = " [Y/n]" if default else " [y/N]"
    raw  = input(c(f"  ❯ {msg}{hint}: ", "yellow")).strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


# ── PSI severity colours ───────────────────────────────────────────────────────
def psi_color(level: str) -> str:
    return {
        "Major":      "red",
        "Moderate":   "yellow",
        "Minor":      "green",
        "Negligible": "dim",
    }.get(level, "dim")


def psi_bar(psi: float, level: str, width: int = 20) -> str:
    """ASCII progress bar scaled to PSI 0.5 max."""
    filled = min(int((psi / 0.5) * width), width)
    bar    = "█" * filled + "░" * (width - filled)
    clr    = psi_color(level)
    return c(bar, clr) + f"  {c(f'{psi:.4f}', clr, 'bold')}"


# ── load a dataset via DataIQ's Dataset loader ────────────────────────────────
def load_dataset(path: str, label: str) -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    try:
        from dataiq.core.dataset import Dataset
        ds = Dataset(path)
        df = ds.df
        print(c(f"  ✓ {label}: {ds.name}  ({df.shape[0]:,} rows × {df.shape[1]} cols)", "green"))
        return df
    except Exception as e:
        print(c(f"  ✗ Failed to load {label} ({path}): {e}", "red"))
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(report: Dict[str, Any]) -> None:
    """Print the dataset-level drift verdict to the terminal."""
    level = report["drift_level"]
    clr   = psi_color(level)
    print(c(f"\n  Verdict : {report['verdict']}", clr, "bold"))
    print(c(f"  Avg PSI : {report['avg_psi']}", clr))
    print(
        f"  Counts  : "
        f"{c('Major', 'red')}={report['n_major']}  "
        f"{c('Moderate', 'yellow')}={report['n_moderate']}  "
        f"{c('Minor', 'green')}={report['n_minor']}  "
        f"{c('Negligible', 'dim')}={report['n_negligible']}"
    )
    print(f"  Columns : {report['n_cols_analyzed']} analyzed  "
          f"|  ref={report['n_ref']:,} rows  |  new={report['n_new']:,} rows")


def print_column_table(report: Dict[str, Any]) -> None:
    """Print a compact per-column drift table."""
    cols = report.get("columns", [])
    if not cols:
        return

    header_line = (
        f"  {'Column':<22}  {'Type':<12}  {'PSI':<8}  {'Level':<12}  "
        f"{'Stat Sig':<10}  Details"
    )
    print(c(header_line, "dim"))
    print(c("  " + "─" * 82, "dim"))

    for col in cols:
        level = col.get("drift_level") or "—"
        clr   = psi_color(level)
        psi   = col.get("psi")
        psi_s = f"{psi:.4f}" if psi is not None else "  —   "

        # Statistical significance indicator
        if col["type"] == "numeric":
            sig = c("KS ✓", "yellow") if col.get("ks_significant") else c("KS —", "dim")
        else:
            sig = c("χ² ✓", "yellow") if col.get("chi_significant") else c("χ² —", "dim")

        # Extra detail snippet
        detail = ""
        if col["type"] == "numeric" and col.get("mean_shift_pct") is not None:
            shift = col["mean_shift_pct"]
            sc    = "red" if shift > 20 else "yellow" if shift > 10 else "dim"
            detail = c(f"mean shift {shift}%", sc)
        elif col["type"] == "categorical":
            parts = []
            if col.get("n_appeared"):
                parts.append(c(f"+{col['n_appeared']} cats", "green"))
            if col.get("n_disappeared"):
                parts.append(c(f"-{col['n_disappeared']} cats", "red"))
            detail = "  ".join(parts)

        print(
            f"  {c(col['column'][:22], clr):<32}"
            f"  {col['type']:<12}"
            f"  {c(psi_s, clr):<18}"
            f"  {c(level, clr):<22}"
            f"  {sig:<20}"
            f"  {detail}"
        )


def print_missing_drift(report: Dict[str, Any]) -> None:
    rows = report.get("missing_drift", [])
    if not rows:
        print(c("  No significant missing-rate changes.", "green"))
        return
    print(c(f"  {len(rows)} column(s) with missing-rate drift (>3pp):\n", "yellow"))
    for r in rows:
        arrow = c("↑", "red") if r["delta"] > 0 else c("↓", "green")
        sev   = r["severity"]
        sc    = "red" if sev == "high" else "yellow" if sev == "medium" else "dim"
        print(
            f"    {c(r['column'], 'cyan'):<28}"
            f"  ref={r['ref_miss']}%  →  new={r['new_miss']}%"
            f"  {arrow} {c(abs(r['delta']), sc)}pp  [{c(sev, sc)}]"
        )


def print_schema_diff(report: Dict[str, Any]) -> None:
    sc = report.get("schema", {})
    if not sc.get("has_changes"):
        print(c("  No schema changes — columns and dtypes match.", "green"))
        return
    if sc.get("appeared"):
        cols = ", ".join(c(col, "green") for col in sc["appeared"])
        print(f"  {c('✚ New in production:', 'green')} {cols}")
    if sc.get("disappeared"):
        cols = ", ".join(c(col, "red") for col in sc["disappeared"])
        print(f"  {c('✖ Missing from production:', 'red')} {cols}")
    if sc.get("dtype_changed"):
        for d in sc["dtype_changed"]:
            print(
                f"  {c('⚙ Dtype changed:', 'yellow')} {d['column']}  "
                f"{c(d['ref_dtype'], 'red')} → {c(d['new_dtype'], 'green')}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL POST-DRIFT ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_leakage_check(df_ref: pd.DataFrame, target: Optional[str],
                      output_dir: str) -> None:
    section("Leakage Check (Reference Dataset)")
    try:
        from dataiq.core.leakage import LeakageDetector

        corr_str = ask("Correlation threshold for leakage flag", default="0.85")
        try:
            corr = float(corr_str)
        except ValueError:
            corr = 0.85

        ld     = LeakageDetector(df_ref, target=target, corr_threshold=corr)
        result = ld.detect()

        print(c(f"\n  Risk Level : {result['risk_level']}", "bold"))
        print(c(f"  Verdict    : {result['verdict']}", "cyan"))
        print(
            f"  Findings   : {result['n_total']} total  |  "
            f"{c('critical', 'red')}={result['n_critical']}  "
            f"{c('high', 'yellow')}={result['n_high']}  "
            f"{c('medium', 'dim')}={result['n_medium']}"
        )

        if result["findings"]:
            print()
            for f in result["findings"]:
                sev = f["severity"].upper()
                clr = "red" if sev in ("CRITICAL", "HIGH") else "yellow"
                print(f"  {f['icon']}  {c(f'[{sev}]', clr)}  {c(f['column'], 'cyan')}")
                print(c(f"       {f['evidence']}", "dim"))
                print(c(f"       Fix: {f['fix']}", "dim"))
                print()

        # Save leakage JSON
        out_path = Path(output_dir) / "leakage_report.json"
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(c(f"  Leakage report saved → {out_path}", "green"))

    except ImportError as e:
        print(c(f"  ✗ Could not import LeakageDetector: {e}", "red"))


def run_pipeline_export(df_ref: pd.DataFrame, target: Optional[str],
                        output_dir: str) -> None:
    section("Export sklearn Pipeline Code")
    try:
        from dataiq.core.analyzer  import AdvancedAnalyzer
        from dataiq.core.code_gen  import PipelineCodeGenerator

        print(c("  Analysing reference data to build pipeline...", "dim"))
        analysis = AdvancedAnalyzer(df_ref, target=target, label="reference").analyze()

        pt_options = ["Auto-detect", "classification", "regression"]
        print(
            "  Problem type:  "
            + "  ".join(f"[{c(i+1,'cyan','bold')}] {o}" for i, o in enumerate(pt_options))
        )
        raw_pt = ask("Choose", default=1)
        try:
            idx = int(raw_pt) - 1
            if idx < 0 or idx >= len(pt_options):
                idx = 0
        except ValueError:
            idx = 0

        problem_type = None if idx == 0 else pt_options[idx]

        gen  = PipelineCodeGenerator(
            analysis=analysis,
            target=target,
            problem_type=problem_type,
        )
        code = gen.generate()

        out_path = Path(output_dir) / "pipeline.py"
        out_path.write_text(code, encoding="utf-8")
        print(c(f"\n  Generated {len(code.splitlines())} lines of pipeline code.", "green"))
        print(f"  Problem type : {gen.problem_type}")
        print(f"  Numeric cols : {gen.num_cols}")
        print(f"  OHE cols     : {gen.cat_ohe}")
        print(f"  Ordinal cols : {gen.cat_ord}")
        print(f"  Dropped      : {gen.drop_cols}")
        print(f"  Scaler       : {'Robust' if gen.use_robust else 'Standard'}")
        print(f"  Imputer      : {'KNN' if gen.use_knn else 'Median/Mode'}")
        if gen.is_imbalanced:
            print(c(f"  ⚠ Imbalanced target (ratio ≈ {gen.imbalance_ratio:.1f}×)", "yellow"))
        print(c(f"\n  Pipeline code saved → {out_path}", "green"))

    except ImportError as e:
        print(c(f"  ✗ Could not import DataIQ modules: {e}", "red"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DataIQ Drift Monitor — compare train vs production data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap_dedent("""
        Examples
        --------
          python dataiq_drift.py --ref train.csv --new production.csv
          python dataiq_drift.py --ref train.csv --new production.csv --target churn
          python dataiq_drift.py --ref train.csv --new production.csv \\
              --target churn --cutoff 2024-01-01 --output drift_out/ --no-browser
        """),
    )
    parser.add_argument("--ref",        required=True,  help="Reference / training dataset path")
    parser.add_argument("--new",        required=True,  help="New / production dataset path")
    parser.add_argument("--target",     default=None,   help="Target column name (excluded from drift)")
    parser.add_argument("--cutoff",     default=None,   help="Future-data cutoff date (ISO: YYYY-MM-DD)")
    parser.add_argument("--threshold",  default=0.85,   type=float,
                        help="PSI threshold for 'Major' drift (default: 0.85 → uses DriftAnalyzer default)")
    parser.add_argument("--output",     default="drift_output", help="Output directory")
    parser.add_argument("--no-browser", action="store_true",    help="Do not open HTML report in browser")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    header("DataIQ — Drift Monitor")

    # ── 1. Load datasets ──────────────────────────────────────────────
    section("Loading Datasets")
    df_ref = load_dataset(args.ref, "Reference")
    df_new = load_dataset(args.new, "New")

    print(f"\n  Shared columns : {len(set(df_ref.columns) & set(df_new.columns))}")
    ref_only = set(df_ref.columns) - set(df_new.columns)
    new_only  = set(df_new.columns) - set(df_ref.columns)
    if ref_only:
        print(c(f"  In ref only    : {sorted(ref_only)}", "yellow"))
    if new_only:
        print(c(f"  In new only    : {sorted(new_only)}", "yellow"))

    # ── 2. Target column ─────────────────────────────────────────────
    section("Target Column")
    target = args.target
    if not target:
        print("  Common columns:")
        common = sorted(set(df_ref.columns) & set(df_new.columns))
        for i, col in enumerate(common):
            print(f"    {c(i+1, 'cyan')}. {col}  [{df_ref[col].dtype}]")
        raw = ask("Enter target column name or number (blank = none)", default="")
        if raw:
            try:
                target = common[int(raw) - 1]
            except (ValueError, IndexError):
                target = raw if raw in df_ref.columns else None
    if target:
        print(c(f"  Target: {target}  (excluded from drift analysis)", "green"))
    else:
        print(c("  No target — all columns will be analyzed.", "dim"))

    # ── 3. Future-data cutoff ─────────────────────────────────────────
    section("Future-Data Cutoff")
    cutoff = args.cutoff
    dt_cols = df_ref.select_dtypes(include=["datetime64"]).columns.tolist()
    if dt_cols:
        print(f"  Datetime columns found: {dt_cols}")
        if not cutoff:
            raw_cutoff = ask(
                "Enter cutoff date YYYY-MM-DD to flag future rows (blank = skip)",
                default="",
            )
            cutoff = raw_cutoff if raw_cutoff else None
        if cutoff:
            print(c(f"  Cutoff: {cutoff}", "cyan"))
        else:
            print(c("  No cutoff set — future-data check will use heuristic.", "dim"))
    else:
        print(c("  No datetime columns — future-data check skipped.", "dim"))
        cutoff = None

    # ── 4. Run drift analysis ─────────────────────────────────────────
    section("Running Drift Analysis")
    print(c("  Computing PSI, KS-test, chi-square, schema diff...", "dim"))

    try:
        from dataiq.core.drift import DriftAnalyzer
    except ImportError as e:
        print(c(f"  ✗ Cannot import DriftAnalyzer: {e}", "red"))
        sys.exit(1)

    da     = DriftAnalyzer(df_ref, df_new, target=target)
    report = da.analyze()

    # ── 5. Print results ──────────────────────────────────────────────
    section("Drift Summary")
    print_summary(report)

    section("Per-Column Drift")
    print_column_table(report)

    section("Missing Rate Drift")
    print_missing_drift(report)

    section("Schema Diff")
    print_schema_diff(report)

    # ── 6. Save HTML report ───────────────────────────────────────────
    section("Saving Outputs")
    try:
        from dataiq.core.report_builder import ReportBuilder

        html_path = str(Path(args.output) / "drift_report.html")
        rb        = ReportBuilder(
            analysis_before={},
            drift_result=report,
            dataset_name=Path(args.ref).stem,
        )
        rb.render_drift(html_path)
        print(c(f"  HTML report → {html_path}", "green"))

        if not args.no_browser:
            import webbrowser
            try:
                webbrowser.open(f"file://{os.path.abspath(html_path)}")
            except Exception:
                pass

    except Exception as e:
        print(c(f"  ✗ Could not generate HTML report: {e}", "red"))

    # ── 7. Save JSON ──────────────────────────────────────────────────
    json_path = Path(args.output) / "drift_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    print(c(f"  JSON result  → {json_path}", "green"))

    # ── 8. Optional leakage check ─────────────────────────────────────
    if confirm("\nRun leakage check on the reference dataset?", default=False):
        run_leakage_check(df_ref, target, args.output)

    # ── 9. Optional pipeline code export ─────────────────────────────
    if confirm("Export sklearn pipeline code for the reference dataset?", default=False):
        run_pipeline_export(df_ref, target, args.output)

    # ── Done ──────────────────────────────────────────────────────────
    header("Drift Analysis Complete!")
    level = report["drift_level"]
    clr   = psi_color(level)
    print(f"  Dataset   : {c(Path(args.ref).stem, 'cyan', 'bold')} vs {c(Path(args.new).stem, 'cyan', 'bold')}")
    print(f"  Result    : {c(level, clr, 'bold')}  (avg PSI = {report['avg_psi']})")
    print(f"  Output    : {c(args.output, 'dim')}")
    print()


# ── textwrap.dedent alias (avoids import at top for cleanliness) ──────────────
def textwrap_dedent(s: str) -> str:
    import textwrap
    return textwrap.dedent(s)


if __name__ == "__main__":
    main()