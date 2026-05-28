"""
report_builder.py — ReportBuilder
====================================
Renders HTML reports for DataIQ.

Original method (preserved exactly):
    render(output_path)       — single or before/after comparison report

New methods added:
    render_drift(output_path) — PSI + KS drift report

Both methods:
  - JSON-safe serialise the payload  (</  →  <\\/ prevents script tag break)
  - Replace all __PLACEHOLDER__ tokens in the template
  - Create parent directories as needed
  - Return the resolved absolute path string
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


class ReportBuilder:
    """
    Render DataIQ HTML reports from analysis dicts.

    Parameters
    ----------
    analysis_before : dict               — before / single analysis result
    analysis_after  : dict, optional     — after analysis result (compare mode)
    drift_result    : dict, optional     — drift analysis result (drift mode)
    dataset_name    : str                — display name shown in the report header
    """

    def __init__(
        self,
        analysis_before: Dict[str, Any],
        analysis_after:  Optional[Dict[str, Any]] = None,
        drift_result:    Optional[Dict[str, Any]] = None,
        dataset_name:    str = "dataset",
    ):
        self.before = analysis_before
        self.after  = analysis_after
        self.drift  = drift_result
        self.name   = dataset_name
        self.ts     = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ══════════════════════════════════════════════════════════════════
    # ORIGINAL METHOD  (preserved exactly, only docstring improved)
    # ══════════════════════════════════════════════════════════════════

    def render(self, output_path: str) -> str:
        """
        Render a single EDA report or a before/after comparison report.

        Mode is determined automatically:
          - analysis_after is None  → single report
          - analysis_after provided → comparison report

        Parameters
        ----------
        output_path : str — file path for the output HTML

        Returns
        -------
        str — resolved absolute path to the written file
        """
        from ..templates.html_template import get_template, ML_KNOWLEDGE

        template = get_template()

        # Prepare payload
        if self.after:
            payload = {"before": self.before, "after": self.after}
            mode    = "compare"
        else:
            payload = self.before
            mode    = "single"

        # Safe JSON — prevents </script> from breaking the page
        data_json = json.dumps(payload, default=str).replace("</", "<\\/")

        # Replace all placeholders
        html = (
            template
            .replace("__DATASET_NAME__",  self.name)
            .replace("__GENERATED_AT__",  self.ts)
            .replace("__ANALYSIS_DATA__", data_json)
            .replace("__REPORT_MODE__",   mode)
            .replace("__ML_KNOWLEDGE__",  ML_KNOWLEDGE)
        )

        return self._write(html, output_path)

    # ══════════════════════════════════════════════════════════════════
    # NEW METHOD — Drift Report
    # ══════════════════════════════════════════════════════════════════

    def render_drift(self, output_path: str) -> str:
        """
        Render a drift report comparing reference vs new dataset.

        Requires drift_result to be passed to __init__.

        Parameters
        ----------
        output_path : str — file path for the output HTML

        Returns
        -------
        str — resolved absolute path to the written file
        """
        from ..templates.drift_template import get_drift_template

        template  = get_drift_template()
        data_json = json.dumps(self.drift or {}, default=str).replace("</", "<\\/")

        html = (
            template
            .replace("__DATASET_NAME__", self.name)
            .replace("__GENERATED_AT__", self.ts)
            .replace("__DRIFT_DATA__",   data_json)
        )

        return self._write(html, output_path)

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL HELPER
    # ══════════════════════════════════════════════════════════════════

    def _write(self, html: str, output_path: str) -> str:
        """Write html to output_path, create parent dirs, return abs path."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        return str(out.resolve())