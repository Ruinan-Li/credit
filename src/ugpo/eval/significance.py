"""Hierarchical paired bootstrap confidence intervals and Holm correction (Table 13).

Implements the statistical reliability protocol from Appendix I.8:
resampling preserves matched examples within each random seed, estimates the
95% confidence interval of paired score differences Delta, and applies
Holm-Bonferroni step-down correction across benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np


@dataclass
class PairedBootstrapResult:
    """Paired significance summary for one benchmark."""

    mean_diff: float
    ci_low: float
    ci_high: float
    raw_p_value: float
    corrected_p_value: float = 0.0


def hierarchical_paired_bootstrap(
    scores_method: Sequence[Sequence[float]],
    scores_baseline: Sequence[Sequence[float]],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Compute hierarchical paired bootstrap 95% CI and two-sided p-value.

    Args:
        scores_method: Nested array of shape [num_seeds, num_examples] for UGPO.
        scores_baseline: Nested array of identical shape for the baseline method.
        n_resamples: Number of bootstrap replicates.
        alpha: Significance level (0.05 -> 95% CI).
        seed: Random seed for reproducible resampling.
    """
    arr_a = np.asarray(scores_method, dtype=np.float64)
    arr_b = np.asarray(scores_baseline, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}.")
    if arr_a.ndim == 1:
        arr_a = arr_a[None, :]
        arr_b = arr_b[None, :]

    diffs = arr_a - arr_b  # [num_seeds, num_examples]
    num_seeds, num_examples = diffs.shape
    observed_diff = float(diffs.mean())

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        # Level 1: resample seeds with replacement; Level 2: resample matched examples within each seed.
        seed_indices = rng.integers(0, num_seeds, size=num_seeds)
        ex_indices = rng.integers(0, num_examples, size=(num_seeds, num_examples))
        sampled = diffs[seed_indices[:, None], ex_indices]
        boot_means[b] = sampled.mean()

    ci_low = float(np.quantile(boot_means, alpha / 2.0))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    # Two-sided bootstrap p-value testing H0: median paired difference == 0.
    p_le_zero = float((np.sum(boot_means <= 0.0) + 1) / (n_resamples + 1))
    p_ge_zero = float((np.sum(boot_means >= 0.0) + 1) / (n_resamples + 1))
    raw_p = min(1.0, 2.0 * min(p_le_zero, p_ge_zero))

    return PairedBootstrapResult(
        mean_diff=observed_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        raw_p_value=raw_p,
        corrected_p_value=raw_p,
    )


def holm_bonferroni_correction(
    results_by_benchmark: Mapping[str, PairedBootstrapResult],
) -> Dict[str, PairedBootstrapResult]:
    """Apply Holm step-down family-wise error correction across benchmarks."""
    keys = list(results_by_benchmark.keys())
    m = len(keys)
    order = sorted(range(m), key=lambda i: results_by_benchmark[keys[i]].raw_p_value)

    corrected: Dict[str, PairedBootstrapResult] = {}
    running_max = 0.0
    for rank, idx in enumerate(order):
        key = keys[idx]
        res = results_by_benchmark[key]
        adj = min(1.0, res.raw_p_value * (m - rank))
        running_max = max(running_max, adj)
        corrected[key] = PairedBootstrapResult(
            mean_diff=res.mean_diff,
            ci_low=res.ci_low,
            ci_high=res.ci_high,
            raw_p_value=res.raw_p_value,
            corrected_p_value=float(running_max),
        )
    return corrected
