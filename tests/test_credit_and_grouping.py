"""Unit tests for responsibility routing, item-use grouping, channel-wise normalization, and delayed-use replay."""

from __future__ import annotations

import math

import numpy as np

from ugpo.config import GroupingConfig, ReplayConfig, RewardConfig
from ugpo.credit.grouping import (
    FrozenGroupingMaps,
    ItemUseGroupBuilder,
    compute_exact_state_coverage,
    theoretical_exact_state_bound,
)
from ugpo.credit.normalization import (
    normalize_and_combine_advantages,
    verify_within_stratum_order,
)
from ugpo.credit.replay import DelayedUseReplayQueue
from ugpo.credit.routing import ResponsibilityRouter
from ugpo.trace.schema import EventType, RolloutTrace, TokenSpanData, WRUEvent


def _make_event(
    event_id: str,
    event_type: EventType,
    item_id: str | None = None,
    step: int = 1,
    session: int = 0,
    request_id: str | None = None,
    traj_id: str = "t1",
    input_group_id: str = "ep1",
    store_hash: str = "h1",
    item_bucket: int = 0,
    task_stratum: tuple = ("hotpotqa", "bridge", 1, 2),
    item_tokens: int = 20,
    is_retrieved: bool = False,
    is_cited: bool = False,
    is_grounded: bool = False,
    is_valid_citation: bool = False,
    task_reward_raw: float = 0.0,
    is_successful_answer: bool = False,
) -> WRUEvent:
    return WRUEvent(
        event_id=event_id,
        event_type=event_type,
        item_id=item_id,
        step=step,
        session=session,
        request_id=request_id,
        trajectory_id=traj_id,
        input_group_id=input_group_id,
        store_hash=store_hash,
        item_bucket=item_bucket,
        task_stratum=task_stratum,
        item_tokens=item_tokens,
        write_step=step,
        write_session=session,
        is_retrieved=is_retrieved,
        is_cited=is_cited,
        is_grounded=is_grounded,
        is_valid_citation=is_valid_citation,
        task_reward_raw=task_reward_raw,
        is_successful_answer=is_successful_answer,
    )


def test_responsibility_routing_matches_table_15_and_eq_7() -> None:
    router = ResponsibilityRouter(RewardConfig())

    # Case 1 (Table 15 row 1): grounded evidence followed by an unsuccessful answer.
    # Originating write gets task=0 (protected from answer failure), positive use=1, and storage cost.
    w1 = _make_event("w1", EventType.WRITE, item_id="m1", item_tokens=100, is_retrieved=True, is_cited=True, is_grounded=True)
    r1 = _make_event("r1", EventType.RETRIEVE, item_id="m1", request_id="q1", item_tokens=100, is_retrieved=True, is_cited=True, is_grounded=True)
    c1 = _make_event("c1", EventType.CITE, item_id="m1", request_id="q1", is_retrieved=True, is_cited=True, is_grounded=True, is_valid_citation=True)
    o1 = _make_event("o1", EventType.OUTCOME, request_id="q1", task_reward_raw=-1.0, is_successful_answer=False)
    trace_fail = RolloutTrace("t_fail", "ep1", events=[w1, r1, c1, o1], terminal_reward=-1.0)
    router.route_trace(trace_fail)

    assert w1.reward_task == 0.0
    assert w1.reward_use == 1.0
    assert math.isclose(w1.reward_cost, -1e-5 * 100)
    assert o1.reward_task == -1.0

    # Case 2 (Table 15 row 4): unsupported citation of a live item.
    # Citation incurs -lambda_spam, while write receives task=0 and use=0 (not penalized on use).
    w2 = _make_event("w2", EventType.WRITE, item_id="m2", item_tokens=50, is_retrieved=True, is_cited=True, is_grounded=False)
    r2 = _make_event("r2", EventType.RETRIEVE, item_id="m2", request_id="q2", item_tokens=50, is_retrieved=True, is_cited=True, is_grounded=False)
    c2 = _make_event("c2", EventType.CITE, item_id="m2", request_id="q2", is_retrieved=True, is_cited=True, is_grounded=False, is_valid_citation=True)
    o2 = _make_event("o2", EventType.OUTCOME, request_id="q2", task_reward_raw=1.0, is_successful_answer=True)
    trace_unsup = RolloutTrace("t_unsup", "ep1", events=[w2, r2, c2, o2], terminal_reward=1.0)
    router.route_trace(trace_unsup)

    assert w2.reward_task == 0.0
    assert w2.reward_use == 0.0
    assert math.isclose(c2.reward_use, -0.5)
    assert c2.reward_task == 0.0
    assert math.isclose(r2.reward_cost, -0.02 - 1e-5 * 50)


def test_exact_state_coverage_decay_vs_item_use_grouping() -> None:
    # Diverged memory stores produce distinct store hashes -> exact-state coverage collapses to 0.
    traces = []
    all_events = []
    for idx in range(4):
        w = _make_event(
            f"w_{idx}",
            EventType.WRITE,
            item_id=f"m_{idx}",
            traj_id=f"traj_{idx}",
            store_hash=f"unique_store_hash_{idx}",
            item_bucket=3,
            is_retrieved=True,
            is_cited=True,
            is_grounded=(idx % 2 == 0),
        )
        w.reward_task = 1.0 if (idx % 2 == 0) else 0.0
        w.reward_use = 1.0 if (idx % 2 == 0) else 0.0
        w.reward_cost = -0.0002 * (idx + 1)
        traces.append(RolloutTrace(f"traj_{idx}", "ep1", events=[w]))
        all_events.append(w)

    assert compute_exact_state_coverage(traces) == 0.0
    assert theoretical_exact_state_bound(batch_size=8, num_writes=10, gamma=0.5) < 0.01

    builder = ItemUseGroupBuilder(GroupingConfig())
    groups = builder.build_groups(all_events)
    diag = builder.group_diagnostics(groups)
    # Item-use key groups all 4 compatible writes despite diverged store hashes.
    assert diag["non_singleton_coverage"] == 1.0
    assert diag["effective_group_size"] == 4.0


def test_channel_wise_normalization_preserves_unit_scale_and_proposition_1() -> None:
    cfg_cw = RewardConfig(channel_wise_normalization=True)
    cfg_joint = RewardConfig(channel_wise_normalization=False)

    def make_group_events() -> list[WRUEvent]:
        evs = []
        raw_triples = [
            (1.0, 1.0, -0.0001),
            (0.8, 0.0, -0.0002),
            (0.2, 1.0, -0.0004),
            (0.0, 0.0, -0.0008),
        ]
        for idx, (rt, ru, rc) in enumerate(raw_triples):
            ev = _make_event(f"ev_{idx}", EventType.WRITE, item_id=f"m_{idx}", item_bucket=1, is_retrieved=True)
            ev.reward_task = rt
            ev.reward_use = ru
            ev.reward_cost = rc
            evs.append(ev)
        return evs

    builder = ItemUseGroupBuilder(GroupingConfig())

    groups_cw = builder.build_groups(make_group_events())
    stats_cw = normalize_and_combine_advantages(groups_cw, cfg_cw)
    assert math.isclose(stats_cw.task_channel_sd, 1.0, abs_tol=1e-5)
    assert math.isclose(stats_cw.use_channel_sd, 1.0, abs_tol=1e-5)
    assert math.isclose(stats_cw.cost_channel_sd, 1.0, abs_tol=1e-5)
    for grp in groups_cw:
        assert verify_within_stratum_order(grp, channel="task")
        assert verify_within_stratum_order(grp, channel="use")
        assert verify_within_stratum_order(grp, channel="cost")

    # Under joint normalization, the tiny cost channel loses unit scale.
    groups_joint = builder.build_groups(make_group_events())
    stats_joint = normalize_and_combine_advantages(groups_joint, cfg_joint)
    assert stats_joint.cost_channel_sd < 0.05


def test_delayed_use_replay_expiry_and_ratio_cutoff() -> None:
    q_cfg = ReplayConfig(
        enabled=True,
        replay_ratio=1.0,
        max_replay_age_sessions=32,
        importance_ratio_min=0.5,
        importance_ratio_max=2.0,
    )
    queue = DelayedUseReplayQueue(q_cfg, seed=7)
    builder = ItemUseGroupBuilder(GroupingConfig())

    origin_old = _make_event("w_old", EventType.WRITE, item_id="m_old", step=1, session=0, is_retrieved=True)
    origin_old.token_span = TokenSpanData(
        input_ids=[1, 2, 3, 4],
        prompt_length=2,
        responsible_mask=[True, True],
        behavior_logprobs=[-1.0, -1.0],
    )
    origin_fresh = _make_event("w_fresh", EventType.WRITE, item_id="m_fresh", step=1, session=20, is_retrieved=True)
    origin_fresh.token_span = TokenSpanData(
        input_ids=[1, 2, 3, 4],
        prompt_length=2,
        responsible_mask=[True, True],
        behavior_logprobs=[-1.0, -1.0],
    )
    queue.register_write(origin_old)
    queue.register_write(origin_fresh)

    key = builder.compute_item_use_key(origin_old)
    queue.enqueue_delayed_pair(origin_old, use_event=None, grouping_key=key, current_session=0)
    queue.enqueue_delayed_pair(origin_fresh, use_event=None, grouping_key=key, current_session=20)

    # At session 40, origin_old (age 40 > 32) expires while origin_fresh (age 20 <= 32) stays eligible.
    current_ev = _make_event("w_cur", EventType.WRITE, item_id="m_cur", step=1, session=40, is_retrieved=True)
    sampled, diag = queue.sample_matching_replay(
        current_events=[current_ev],
        key_fn=builder.compute_item_use_key,
        current_session=40,
        logprob_evaluator=lambda _ev: [-0.8, -1.1],  # ratios exp(0.2)=1.22, exp(-0.1)=0.90 in [0.5, 2.0]
    )
    assert diag["expired"] == 1
    assert diag["ratio_discarded"] == 0
    assert len(sampled) == 1

    # Drift beyond [0.5, 2.0] (log_ratio = 1.0 -> ratio = 2.718 > 2.0) discards entry before clipping.
    sampled2, diag2 = queue.sample_matching_replay(
        current_events=[current_ev],
        key_fn=builder.compute_item_use_key,
        current_session=41,
        logprob_evaluator=lambda _ev: [0.0, -1.0],
    )
    assert diag2["ratio_discarded"] == 1
    assert len(sampled2) == 0


def test_frozen_grouping_maps_save_load_and_immutability(tmp_path) -> None:
    rng = np.random.default_rng(123)
    embs = rng.normal(size=(40, 16)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    maps = FrozenGroupingMaps(GroupingConfig(num_item_buckets=8))
    maps.fit_item_buckets(embs, seed=42)
    maps.fit_task_strata(context_lengths=[100, 500, 1200, 4000, 9000], ref_entropies=[0.2, 0.6, 1.0, 1.4, 2.1])
    maps.freeze()

    maps.save(tmp_path)
    reloaded = FrozenGroupingMaps.load(tmp_path, GroupingConfig(num_item_buckets=8))
    assert reloaded.is_frozen
    assert reloaded.centroids is not None and reloaded.centroids.shape == (8, 16)
    assert reloaded.assign_item_bucket("wikipedia", embs[0]) == maps.assign_item_bucket("wikipedia", embs[0])
    assert reloaded.assign_delay_bin(0) == "delay_bin_0"
    assert reloaded.assign_delay_bin(8) == "delay_bin_4"
    assert reloaded.assign_delay_bin(None) == "NO-USE"
