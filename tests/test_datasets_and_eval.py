"""Integration tests on the bundled HotpotQA and LoCoMo datasets and evaluation suite."""

from __future__ import annotations

from pathlib import Path

from ugpo.data.loaders import load_hotpotqa, load_locomo
from ugpo.eval.metrics import (
    compute_f1,
    compute_four_task_macro_average,
    compute_lifecycle_metrics,
    compute_locomo_score,
)
from ugpo.eval.provenance import evaluate_controlled_provenance
from ugpo.eval.significance import hierarchical_paired_bootstrap, holm_bonferroni_correction
from ugpo.trace.schema import EventType, RolloutTrace, WRUEvent


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_bundled_hotpotqa_dataset() -> None:
    path = REPO_ROOT / "data" / "hotpotqa" / "hotpot_dev_distractor.parquet"
    assert path.exists()

    episodes = load_hotpotqa(path, limit=10)
    assert len(episodes) == 10
    first = episodes[0]
    assert first.benchmark == "hotpotqa"
    assert len(first.stream_units) == 10
    assert len(first.requests) == 1
    assert len(first.requests[0].gold_supports) >= 2
    assert first.requests[0].question != ""
    assert first.requests[0].answer != ""


def test_load_bundled_locomo_dataset() -> None:
    path = REPO_ROOT / "data" / "locomo" / "locomo10.json"
    assert path.exists()

    episodes = load_locomo(path, limit=3)
    assert len(episodes) == 3
    first = episodes[0]
    assert first.benchmark == "locomo"
    assert len(first.stream_units) > 100
    assert len(first.requests) > 50
    assert any(len(req.gold_supports) > 0 for req in first.requests)
    # Session chronology must be non-decreasing across streamed turns.
    sessions = [u.session_id for u in first.stream_units]
    assert sessions == sorted(sessions)


def test_task_metrics_lifecycle_provenance_and_significance() -> None:
    # Task scoring checks including inline citation stripping.
    assert compute_f1("Scott Derrickson [mem_000001]", "Scott Derrickson") == 1.0
    assert compute_locomo_score("7 May 2023 [mem_000004]", "7 May 2023", category=2) == 1.0
    assert compute_locomo_score("No information available in memory.", "decoy", category=5) == 1.0
    macro = compute_four_task_macro_average(
        {"hotpotqa": 73.6, "2wiki": 70.2, "locomo": 61.8, "longmemeval": 68.7}
    )
    assert round(macro, 2) == 68.58

    # Controlled provenance P_q alignment against real HotpotQA gold supports.
    ep = load_hotpotqa(REPO_ROOT / "data" / "hotpotqa" / "hotpot_dev_distractor.parquet", limit=1)[0]
    gold = ep.requests[0].gold_supports
    assert len(gold) >= 1

    w_gold = WRUEvent(
        event_id="w_gold",
        event_type=EventType.WRITE,
        item_id="mem_000001",
        step=1,
        operation="ADD",
        item_tokens=30,
        is_grounded=True,
        reward_task=1.0,
        reward_use=1.0,
        metadata={
            "source_doc_id": gold[0].source_doc_id,
            "char_start": gold[0].char_start,
            "char_end": gold[0].char_end,
        },
    )
    w_distractor = WRUEvent(
        event_id="w_dist",
        event_type=EventType.WRITE,
        item_id="mem_000002",
        step=2,
        operation="ADD",
        item_tokens=45,
        is_grounded=False,
        reward_task=0.0,
        reward_use=0.0,
        metadata={"source_doc_id": "Unrelated Document", "char_start": 0, "char_end": 10},
    )
    r_ev = WRUEvent(
        event_id="r_1",
        event_type=EventType.RETRIEVE,
        item_id="mem_000001",
        step=3,
        request_id="q0",
        item_tokens=30,
        is_retrieved=True,
        is_cited=True,
        is_grounded=True,
    )
    c_ev = WRUEvent(
        event_id="c_1",
        event_type=EventType.CITE,
        item_id="mem_000001",
        step=3,
        request_id="q0",
        is_retrieved=True,
        is_cited=True,
        is_grounded=True,
    )
    trace = RolloutTrace(
        trajectory_id="t0",
        input_group_id=ep.episode_id,
        events=[w_gold, w_distractor, r_ev, c_ev],
        terminal_reward=1.0,
        store_tokens_at_end=75,
        serving_latency_ms=875.0,
    )

    prov = evaluate_controlled_provenance([trace], {ep.episode_id: [gold[0]]})
    assert prov.precision_pct == 100.0
    assert prov.recall_pct == 100.0

    life = compute_lifecycle_metrics([trace])
    assert life.write_utilization_pct == 50.0
    assert life.retrieval_precision_pct == 100.0
    assert life.unsupported_citations_pct == 0.0
    assert life.pollution_tokens == 45.0

    # Hierarchical paired bootstrap + Holm correction (Table 13).
    ugpo_scores = [[73.5, 74.0, 73.8, 73.2, 74.1] for _ in range(5)]
    base_scores = [[72.5, 73.0, 72.7, 72.2, 73.1] for _ in range(5)]
    boot = hierarchical_paired_bootstrap(ugpo_scores, base_scores, n_resamples=1000, seed=42)
    assert boot.ci_low > 0.0
    corrected = holm_bonferroni_correction({"hotpotqa": boot})
    assert corrected["hotpotqa"].corrected_p_value < 0.05
