"""
Evaluation module - computes inter-rater agreement metrics
between LLM annotations and human ground truth.

**Backward-compatibility shim.**  All metric functions now live in
``src.pipeline.metrics``.  This module re-exports them so that
existing Phase 1 scripts (``run_phase1.py``, ``run_task2_qlevel.py``,
etc.) continue to work unchanged.
"""

# Re-export every public name from the canonical location
from src.pipeline.metrics import (  # noqa: F401
    compute_cohens_kappa,
    compute_krippendorff_alpha,
    compute_krippendorff_alpha_multi,
    compute_exact_match,
    compute_within_one,
    compute_per_dimension_f1,
    compute_confusion_matrix,
    compute_expected_calibration_error,
    evaluate_full,
    evaluate_human_agreement,
    print_comparison_table,
    _interpret_kappa,
)
