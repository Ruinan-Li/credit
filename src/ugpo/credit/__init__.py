"""Responsibility routing, item-use grouping, channel-wise normalization, and delayed-use replay."""

from ugpo.credit.grouping import (
    ComparisonGroup,
    FrozenGroupingMaps,
    ItemUseGroupBuilder,
    compute_exact_state_coverage,
    theoretical_exact_state_bound,
)
from ugpo.credit.normalization import (
    ChannelNormalizationStats,
    normalize_and_combine_advantages,
    verify_within_stratum_order,
)
from ugpo.credit.replay import DelayedUseReplayEntry, DelayedUseReplayQueue
from ugpo.credit.routing import ResponsibilityMasks, ResponsibilityRouter

__all__ = [
    "ChannelNormalizationStats",
    "ComparisonGroup",
    "DelayedUseReplayEntry",
    "DelayedUseReplayQueue",
    "FrozenGroupingMaps",
    "ItemUseGroupBuilder",
    "ResponsibilityMasks",
    "ResponsibilityRouter",
    "compute_exact_state_coverage",
    "normalize_and_combine_advantages",
    "theoretical_exact_state_bound",
    "verify_within_stratum_order",
]
