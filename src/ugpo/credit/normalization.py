"""Channel-wise group-relative advantage normalization (Equations 9, 12, and 18).

Separately normalizes the task, use, and cost reward channels within each
comparison group K before linear aggregation, preserving sparse channel
variation that collapses under joint normalization (Section 3.4, Table 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ugpo.config import RewardConfig
from ugpo.credit.grouping import ComparisonGroup
from ugpo.trace.schema import EventType, WRUEvent


@dataclass
class ChannelNormalizationStats:
    """Post-normalization scale diagnostics matching Table 8."""

    task_channel_sd: float
    use_channel_sd: float
    cost_channel_sd: float
    usable_events: int
    zero_variance_groups: int
    singleton_groups: int


def _normalize_channel_values(
    values: Sequence[float],
    eps: float = 1e-8,
) -> tuple[List[float], bool]:
    """Compute (R_e^c - mu_K^c) / sigma_K^c for one group and channel (Equation 9).

    Returns (normalized_values, had_positive_variance). Groups with fewer than
    two events or zero variance contribute zero local advantage (Section 3.3).
    """
    n = len(values)
    if n < 2:
        return [0.0] * n, False
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=0))
    if std <= eps:
        return [0.0] * n, False
    mean = float(arr.mean())
    normed = ((arr - mean) / std).tolist()
    return normed, True


def normalize_and_combine_advantages(
    groups: Sequence[ComparisonGroup],
    config: RewardConfig,
    min_group_size: int = 2,
) -> ChannelNormalizationStats:
    """Apply NORMALIZEREWARDS and COMBINEADVANTAGES (Algorithm 1 lines 11-12).

    Mutates `adv_task`, `adv_use`, `adv_cost`, and `advantage` on every WRUEvent
    inside `groups`.
    """
    singleton_groups = 0
    zero_variance_groups = 0
    usable_events = 0

    eligible_task_advs: List[float] = []
    eligible_use_advs: List[float] = []
    eligible_cost_advs: List[float] = []

    for group in groups:
        evs = group.events
        if len(evs) < min_group_size:
            singleton_groups += 1
            for ev in evs:
                ev.adv_task = 0.0
                ev.adv_use = 0.0
                ev.adv_cost = 0.0
                ev.advantage = 0.0
            continue

        task_vals = [ev.reward_task for ev in evs]
        use_vals = [ev.reward_use for ev in evs]
        cost_vals = [ev.reward_cost for ev in evs]

        if group.group_kind == "answer" or all(ev.event_type == EventType.OUTCOME for ev in evs):
            # Answer tokens use same-input rollout grouping on task return (Equation 2).
            adv_t, ok_t = _normalize_channel_values(task_vals, eps=config.eps)
            if not ok_t:
                zero_variance_groups += 1
            else:
                usable_events += len(evs)
                eligible_task_advs.extend(adv_t)
            for ev, at in zip(evs, adv_t):
                ev.adv_task = float(at)
                ev.adv_use = 0.0
                ev.adv_cost = 0.0
                ev.advantage = float(at)
            continue

        if config.channel_wise_normalization:
            # Independent normalization per channel before aggregation (Equations 9 & 12).
            adv_t, ok_t = _normalize_channel_values(task_vals, eps=config.eps)
            adv_u, ok_u = _normalize_channel_values(use_vals, eps=config.eps)
            adv_c, ok_c = _normalize_channel_values(cost_vals, eps=config.eps)

            if not (ok_t or ok_u or ok_c):
                zero_variance_groups += 1
            else:
                usable_events += len(evs)

            if ok_t:
                eligible_task_advs.extend(adv_t)
            if ok_u:
                eligible_use_advs.extend(adv_u)
            if ok_c:
                eligible_cost_advs.extend(adv_c)

            for ev, at, au, ac in zip(evs, adv_t, adv_u, adv_c):
                ev.adv_task = float(at)
                ev.adv_use = float(au)
                ev.adv_cost = float(ac)
                ev.advantage = float(
                    config.alpha_task * at
                    + config.alpha_use * au
                    + config.alpha_cost * ac
                )
        else:
            # Joint normalization ablation (Table 3 & Table 8): normalize weighted sum once.
            combined_raw = [
                config.alpha_task * rt + config.alpha_use * ru + config.alpha_cost * rc
                for rt, ru, rc in zip(task_vals, use_vals, cost_vals)
            ]
            arr_sum = np.asarray(combined_raw, dtype=np.float64)
            std_sum = float(arr_sum.std(ddof=0))
            if std_sum <= config.eps:
                zero_variance_groups += 1
                for ev in evs:
                    ev.adv_task = 0.0
                    ev.adv_use = 0.0
                    ev.adv_cost = 0.0
                    ev.advantage = 0.0
            else:
                usable_events += len(evs)
                mean_sum = float(arr_sum.mean())
                mean_t = float(np.mean(task_vals))
                mean_u = float(np.mean(use_vals))
                mean_c = float(np.mean(cost_vals))
                for ev, rt, ru, rc, rs in zip(evs, task_vals, use_vals, cost_vals, combined_raw):
                    # Scaled contributions under joint normalization expose scale dominance.
                    ev.adv_task = float((rt - mean_t) / std_sum)
                    ev.adv_use = float((ru - mean_u) / std_sum)
                    ev.adv_cost = float((rc - mean_c) / std_sum)
                    ev.advantage = float((rs - mean_sum) / std_sum)
                    eligible_task_advs.append(ev.adv_task)
                    eligible_use_advs.append(ev.adv_use)
                    eligible_cost_advs.append(ev.adv_cost)

    task_sd = float(np.std(eligible_task_advs)) if eligible_task_advs else 0.0
    use_sd = float(np.std(eligible_use_advs)) if eligible_use_advs else 0.0
    cost_sd = float(np.std(eligible_cost_advs)) if eligible_cost_advs else 0.0

    return ChannelNormalizationStats(
        task_channel_sd=task_sd,
        use_channel_sd=use_sd,
        cost_channel_sd=cost_sd,
        usable_events=usable_events,
        zero_variance_groups=zero_variance_groups,
        singleton_groups=singleton_groups,
    )


def verify_within_stratum_order(group: ComparisonGroup, channel: str = "task") -> bool:
    """Verify Proposition 1 (Equation 18): within-stratum order preservation.

    For any e, e' in K with |K| >= 2 and sigma_K^c > 0:
        A_hat_e^c > A_hat_{e'}^c  iff  R_e^c > R_{e'}^c.
    """
    if group.size < 2:
        return True
    reward_attr = f"reward_{channel}"
    adv_attr = f"adv_{channel}"

    rewards = [float(getattr(ev, reward_attr)) for ev in group.events]
    advs = [float(getattr(ev, adv_attr)) for ev in group.events]
    if float(np.std(rewards)) <= 1e-8:
        return all(abs(a) <= 1e-8 for a in advs)

    for i in range(len(rewards)):
        for j in range(len(rewards)):
            if rewards[i] > rewards[j] + 1e-12 and not (advs[i] > advs[j]):
                return False
    return True
