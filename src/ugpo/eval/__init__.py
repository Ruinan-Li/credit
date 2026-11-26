"""Task metrics, lifecycle serving metrics, controlled provenance, and significance testing."""

from ugpo.eval.metrics import (
    LifecycleMetrics,
    compute_exact_match,
    compute_f1,
    compute_four_task_macro_average,
    compute_lifecycle_metrics,
    compute_locomo_score,
    compute_longmemeval_score,
    compute_task_reward,
    normalize_answer,
)
from ugpo.eval.provenance import (
    AttributionMetrics,
    build_reference_provenance,
    evaluate_controlled_provenance,
)
from ugpo.eval.significance import (
    PairedBootstrapResult,
    hierarchical_paired_bootstrap,
    holm_bonferroni_correction,
)

__all__ = [
    "AttributionMetrics",
    "LifecycleMetrics",
    "PairedBootstrapResult",
    "build_reference_provenance",
    "compute_exact_match",
    "compute_f1",
    "compute_four_task_macro_average",
    "compute_lifecycle_metrics",
    "compute_locomo_score",
    "compute_longmemeval_score",
    "compute_task_reward",
    "evaluate_controlled_provenance",
    "hierarchical_paired_bootstrap",
    "holm_bonferroni_correction",
    "normalize_answer",
]
