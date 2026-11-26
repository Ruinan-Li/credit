#!/usr/bin/env python3
"""Fit and freeze pre-action item-use grouping maps prior to policy optimization.

Implements Section 3.3 and Appendix G:
1. Encodes pre-write source segments using frozen intfloat/e5-large-v2 (or
   lexical hashing when --encoder lexical is specified) and fits K=8 K-Means
   item buckets phi(i).
2. Computes context-length quartiles and optional frozen reference-policy
   greedy entropy quintiles for task strata psi(q) without task labels.
3. Freezes and saves `grouping_maps.json` and `item_bucket_centroids.npy`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ugpo.config import UGPOConfig
from ugpo.credit.grouping import FrozenGroupingMaps
from ugpo.data.loaders import load_benchmark
from ugpo.memory.retriever import E5EmbeddingEncoder, LexicalEmbeddingEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and freeze pre-action UGPO grouping maps before training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ugpo_qwen25_7b.yaml",
        help="Path to UGPO YAML configuration file.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "2wiki", "locomo", "longmemeval", "alfworld", "webshop"],
        help="Benchmark dataset to fit pre-write clusters and strata on.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/hotpotqa/hotpot_dev_distractor.parquet",
        help="Path to dataset file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of episodes used to fit frozen grouping maps.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="e5",
        choices=["e5", "lexical"],
        help="Embedding encoder backend: 'e5' (intfloat/e5-large-v2) or 'lexical'.",
    )
    parser.add_argument(
        "--compute-ref-entropy",
        action="store_true",
        help="Compute reference-policy greedy next-token entropy using optimization.policy_model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/frozen_grouping_maps",
        help="Directory to write frozen grouping_maps.json and item_bucket_centroids.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = UGPOConfig.from_yaml(args.config)

    episodes = load_benchmark(args.benchmark, args.data_path, limit=args.limit)
    source_segments: List[str] = []
    context_lengths: List[int] = []
    ref_entropies: List[float] = []

    for ep in episodes:
        for unit in ep.stream_units:
            source_segments.append(unit.text)
        for req in ep.requests:
            context_lengths.append(req.context_length)

    if args.encoder == "e5":
        encoder = E5EmbeddingEncoder(model_name=cfg.memory.embedding_model)
    else:
        encoder = LexicalEmbeddingEncoder(dim=512)

    embeddings = encoder.encode_passages(source_segments)

    if args.compute_ref_entropy:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ugpo.policy.agent import MemoryPolicyAgent

        tokenizer = AutoTokenizer.from_pretrained(cfg.optimization.policy_model)
        dtype = torch.bfloat16 if (cfg.optimization.bf16 and torch.cuda.is_available()) else torch.float32
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.optimization.policy_model,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        agent = MemoryPolicyAgent(model=ref_model, tokenizer=tokenizer, ref_model=ref_model)
        for ep in episodes:
            for req in ep.requests:
                prompt = agent.build_answer_prompt(req.question, retrieved=[])
                ref_entropies.append(agent.compute_reference_entropy(prompt))

    maps = FrozenGroupingMaps(cfg.grouping)
    maps.fit_item_buckets(embeddings, seed=cfg.optimization.seed)
    maps.fit_task_strata(context_lengths=context_lengths, ref_entropies=ref_entropies)
    maps.freeze()

    out_dir = Path(args.output_dir)
    maps.save(out_dir)
    print(
        f"Saved frozen grouping maps ({cfg.grouping.num_item_buckets} clusters, "
        f"{len(source_segments)} source segments) to {out_dir}"
    )


if __name__ == "__main__":
    main()
