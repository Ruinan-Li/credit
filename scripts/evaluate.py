#!/usr/bin/env python3
"""Evaluate a trained UGPO memory policy on task, lifecycle, and provenance metrics.

Reports:
- Benchmark task score (Table 1: HotpotQA F1, 2Wiki F1, LoCoMo score, LongMemEval score).
- Lifecycle and serving metrics (Table 6: write utilization, retrieval precision,
  unsupported citations, pollution tokens, store size, retrieved tokens/request, latency).
- Group-formation coverage diagnostics (Table 7: exact-state vs item-use coverage).
- Post-hoc controlled provenance attribution precision and recall (Tables 3 and 10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ugpo.config import UGPOConfig
from ugpo.credit.grouping import FrozenGroupingMaps, ItemUseGroupBuilder, compute_exact_state_coverage
from ugpo.credit.normalization import normalize_and_combine_advantages
from ugpo.credit.routing import ResponsibilityRouter
from ugpo.data.loaders import load_benchmark
from ugpo.eval.metrics import compute_lifecycle_metrics, compute_task_reward
from ugpo.eval.provenance import evaluate_controlled_provenance
from ugpo.memory.retriever import E5EmbeddingEncoder, LexicalEmbeddingEncoder
from ugpo.policy.trainer import UGPOTrainer
from ugpo.trace.grounding import GroundedUseValidator
from ugpo.trace.schema import RolloutTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a UGPO checkpoint on memory-agent benchmarks.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ugpo_qwen25_7b.yaml",
        help="Path to UGPO configuration YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained policy checkpoint (defaults to config optimization.policy_model).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="locomo",
        choices=["hotpotqa", "2wiki", "locomo", "longmemeval", "alfworld", "webshop"],
        help="Evaluation benchmark.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/locomo/locomo10.json",
        help="Path to evaluation dataset file.",
    )
    parser.add_argument(
        "--grouping-maps-dir",
        type=str,
        default=None,
        help="Optional directory with frozen grouping maps.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="e5",
        choices=["e5", "lexical"],
        help="Embedding encoder backend.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of episodes to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/eval_report.json",
        help="Destination path for structured evaluation metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = UGPOConfig.from_yaml(args.config)
    model_path = args.checkpoint or cfg.optimization.policy_model

    episodes = load_benchmark(args.benchmark, args.data_path, limit=args.limit)
    if args.encoder == "e5":
        encoder = E5EmbeddingEncoder(model_name=cfg.memory.embedding_model)
    else:
        encoder = LexicalEmbeddingEncoder(dim=512)

    if args.grouping_maps_dir is not None:
        grouping_maps = FrozenGroupingMaps.load(args.grouping_maps_dir, cfg.grouping)
    else:
        grouping_maps = FrozenGroupingMaps(cfg.grouping)
        pre_segments = [u.text for ep in episodes[:256] for u in ep.stream_units]
        grouping_maps.fit_item_buckets(encoder.encode_passages(pre_segments), seed=cfg.optimization.seed)
        grouping_maps.freeze()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if (cfg.optimization.bf16 and torch.cuda.is_available()) else torch.float32
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    policy_model.eval()

    validator = GroundedUseValidator(cfg.grounding)
    router = ResponsibilityRouter(cfg.reward)
    group_builder = ItemUseGroupBuilder(cfg.grouping, grouping_maps=grouping_maps, seed=cfg.optimization.seed)

    def reward_fn(prediction: str, ground_truth: str, metadata: Dict[str, Any]) -> float:
        return compute_task_reward(args.benchmark, prediction, ground_truth, metadata)

    trainer = UGPOTrainer(
        config=cfg,
        policy_model=policy_model,
        ref_model=None,
        tokenizer=tokenizer,
        encoder=encoder,
        validator=validator,
        grouping_maps=grouping_maps,
        reward_fn=reward_fn,
    )

    agentic_mode = args.benchmark in ("alfworld", "webshop")
    traces: List[RolloutTrace] = []
    gold_supports_by_episode = {}

    for ep in episodes:
        trace = trainer.rollout_episode(ep, rollout_idx=0, do_sample=False)
        validator.validate_trace(trace, agentic_mode=agentic_mode)
        router.route_trace(trace)
        traces.append(trace)
        gold_supports_by_episode[ep.episode_id] = [
            g for req in ep.requests for g in req.gold_supports
        ]

    all_events = [ev for tr in traces for ev in tr.events]
    groups = group_builder.build_groups(all_events)
    normalize_and_combine_advantages(groups, cfg.reward, min_group_size=cfg.grouping.min_group_size)
    group_diag = group_builder.group_diagnostics(groups, min_group_size=cfg.grouping.min_group_size)

    task_score_pct = (
        100.0 * sum(tr.terminal_reward for tr in traces) / len(traces) if traces else 0.0
    )
    lifecycle = compute_lifecycle_metrics(traces)
    provenance = evaluate_controlled_provenance(traces, gold_supports_by_episode)
    exact_cov = compute_exact_state_coverage(traces)

    report = {
        "benchmark": args.benchmark,
        "num_episodes": len(traces),
        "task_score_pct": round(task_score_pct, 2),
        "attribution": {
            "precision_pct": round(provenance.precision_pct, 2),
            "recall_pct": round(provenance.recall_pct, 2),
            "f1_pct": round(provenance.f1_pct, 2),
            "precision_by_operation": {
                k: round(v, 2) for k, v in provenance.precision_by_operation.items()
            },
        },
        "lifecycle": {
            "write_utilization_pct": round(lifecycle.write_utilization_pct, 2),
            "retrieval_precision_pct": round(lifecycle.retrieval_precision_pct, 2),
            "unsupported_citations_pct": round(lifecycle.unsupported_citations_pct, 2),
            "pollution_tokens": round(lifecycle.pollution_tokens, 1),
            "store_size_tokens": round(lifecycle.store_size_tokens, 1),
            "retrieved_tokens_per_request": round(lifecycle.retrieved_tokens_per_request, 1),
            "serving_latency_ms": round(lifecycle.serving_latency_ms, 1),
        },
        "grouping": {
            "item_use_non_singleton_coverage": round(group_diag["non_singleton_coverage"], 4),
            "exact_state_non_singleton_coverage": round(exact_cov, 4),
            "effective_group_size": round(group_diag["effective_group_size"], 2),
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
