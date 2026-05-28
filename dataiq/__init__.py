"""
DataIQ — Advanced ML Data Intelligence Suite
=============================================
The ML data library that thinks, not just profiles.

Unique features:
  • ML Readiness Score (MRS) — A/B/C/D/F per column and dataset-wide
  • Leakage Detective      — auto-detects data leakage patterns
  • Drift Analyzer         — PSI + KS-test between two datasets
  • Narrative Engine       — plain-English story for every finding
  • Code Generator         — produces ready-to-run sklearn Pipeline code
  • What-If Simulator      — interactive client-side score explorer in HTML
  • Feature Interactions   — mutual information between all feature pairs
  • Temporal Awareness     — time-series decomposition + concept drift windows
"""
from .core.profiler   import DataIQ
from .core.dataset    import Dataset
from .core.scorer     import ReadinessScorer
from .core.leakage    import LeakageDetector
from .core.drift      import DriftAnalyzer
from .core.code_gen   import PipelineCodeGenerator

__version__ = "1.0.0"
__all__ = ["DataIQ","Dataset","ReadinessScorer","LeakageDetector","DriftAnalyzer","PipelineCodeGenerator"]