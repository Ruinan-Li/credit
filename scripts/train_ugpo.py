#!/usr/bin/env python3
"""Train a long-context memory agent with Use-Grouped Policy Optimization (UGPO).

Implements Algorithm 1 with matched settings from Table 5 and supports all
Table 3 component ablations and Table 14 assignment operators via `--preset`.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from ugpo.config import UGPOConfig
from ugpo.credit.grouping import FrozenGroupingMaps
from ugpo.data.loaders import load_benchmark
from ugpo.eval.metrics import compute_task_reward
from ugpo.memory.retriever import E5EmbeddingEncoder, LexicalEmbeddingEncoder
from ugpo.policy.trainer import UGPOTrainer
from ugpo.trace.grounding import GroundedUseValidator


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], val)
        else:
            merged[key] = val
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a memory agent with UGPO (Algorithm 1).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ugpo_qwen25_7b.yaml",
        help="Path to base UGPO YAML configuration.",
    )
    parser.add_argument(
        "--ablation-file",
        type=str,
        default="configs/ablations.yaml",
        help="Path to YAML file defining ablation/baseline presets.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Optional preset name from configs/ablations.yaml (e.g. outcome_grpo, wo_use_channel).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "2wiki", "locomo", "longmemeval", "alfworld", "webshop"],
        help="Training benchmark name.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/hotpotqa/hotpot_dev_distractor.parquet",
        help="Path to benchmark dataset file.",
    )
    parser.add_argument(
        "--grouping-maps-dir",
        type=str,
        default=None,
        help="Directory containing frozen grouping_maps.json and item_bucket_centroids.npy.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="e5",
        choices=["e5", "lexical"],
        help="Dense retrieval encoder backend.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on loaded episodes.",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=None,
        help="Override optimization.num_updates from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for checkpoints and metrics.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    if args.preset is not None:
        with open(args.ablation_file, "r", encoding="utf-8") as handle:
            presets = (yaml.safe_load(handle) or {}).get("presets", {})
        if args.preset not in presets:
            raise ValueError(f"Unknown preset '{args.preset}'. Available: {sorted(presets.keys())}")
        raw_cfg = _deep_update(raw_cfg, presets[args.preset])
        raw_cfg["experiment_name"] = f"{raw_cfg.get('experiment_name', 'ugpo')}_{args.preset}"

    cfg = UGPOConfig.from_dict(raw_cfg)
    if args.num_updates is not None:
        cfg.optimization.num_updates = args.num_updates
    if args.seed is not None:
        cfg.optimization.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    set_seed(cfg.optimization.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(out_dir / "resolved_config.yaml")

    episodes = load_benchmark(args.benchmark, args.data_path, limit=args.max_episodes)
    if not episodes:
        raise RuntimeError(f"No episodes loaded from {args.data_path}.")

    # Load frozen pre-action grouping maps (or fit once before the first update).
    if args.encoder == "e5":
        encoder = E5EmbeddingEncoder(model_name=cfg.memory.embedding_model)
    else:
        encoder = LexicalEmbeddingEncoder(dim=512)

    if args.grouping_maps_dir is not None:
        grouping_maps = FrozenGroupingMaps.load(args.grouping_maps_dir, cfg.grouping)
    else:
        grouping_maps = FrozenGroupingMaps(cfg.grouping)
        pre_segments = [u.text for ep in episodes[:256] for u in ep.stream_units]
        ctx_lengths = [r.context_length for ep in episodes for r in ep.requests]
        grouping_maps.fit_item_buckets(encoder.encode_passages(pre_segments), seed=cfg.optimization.seed)
        grouping_maps.fit_task_strata(context_lengths=ctx_lengths, ref_entropies=[])
        grouping_maps.freeze()
        grouping_maps.save(out_dir / "frozen_grouping_maps")

    # Load policy backbone pi_theta and frozen reference policy pi_ref.
    tokenizer = AutoTokenizer.from_pretrained(cfg.optimization.policy_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if (cfg.optimization.bf16 and torch.cuda.is_available()) else torch.float32
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.optimization.policy_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if cfg.optimization.gradient_checkpointing and hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable()

    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.optimization.policy_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    validator = GroundedUseValidator(cfg.grounding)

    def reward_fn(prediction: str, ground_truth: str, metadata: Dict[str, Any]) -> float:
        return compute_task_reward(args.benchmark, prediction, ground_truth, metadata)

    trainer = UGPOTrainer(
        config=cfg,
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        encoder=encoder,
        validator=validator,
        grouping_maps=grouping_maps,
        reward_fn=reward_fn,
    )

    episodes_per_step = max(1, cfg.optimization.effective_batch_size // cfg.optimization.rollouts_per_input)
    agentic_mode = args.benchmark in ("alfworld", "webshop")
    log_path = out_dir / "train_metrics.jsonl"

    with open(log_path, "w", encoding="utf-8") as log_file:
        for update_idx in range(cfg.optimization.num_updates):
            batch_episodes = [
                episodes[(update_idx * episodes_per_step + i) % len(episodes)]
                for i in range(episodes_per_step)
            ]
            metrics = trainer.train_step(batch_episodes, agentic_mode=agentic_mode)
            record = {
                "step": metrics.step,
                "loss": metrics.loss,
                "policy_loss": metrics.policy_loss,
                "kl_loss": metrics.kl_loss,
                "mean_task_reward": metrics.mean_task_reward,
                "non_singleton_coverage": metrics.non_singleton_coverage,
                "exact_state_coverage": metrics.exact_state_coverage,
                "effective_group_size": metrics.effective_group_size,
                "task_channel_sd": metrics.task_channel_sd,
                "use_channel_sd": metrics.use_channel_sd,
                "cost_channel_sd": metrics.cost_channel_sd,
                "num_current_events": metrics.num_current_events,
                "num_replay_events": metrics.num_replay_events,
                "trace_processing_ms": metrics.trace_processing_ms,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(
                f"[step {metrics.step}/{cfg.optimization.num_updates}] "
                f"loss={metrics.loss:.4f} reward={metrics.mean_task_reward:.4f} "
                f"cov={metrics.non_singleton_coverage:.3f} "
                f"exact_cov={metrics.exact_state_coverage:.3f} "
                f"replay={metrics.num_replay_events}"
            )

    ckpt_dir = out_dir / "final_checkpoint"
    policy_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Training complete. Checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
