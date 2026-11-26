"""Configuration schemas for Use-Grouped Policy Optimization (UGPO).

All default hyperparameters match the matched experimental setup reported in
Table 5 of "Credit Follows Use: Use-Grouped Policy Optimization for
Long-Context Memory Agent".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass
class MemoryConfig:
    """Persistent external memory store and retrieval configuration."""

    max_context_tokens: int = 131072
    max_memory_tokens: int = 8192
    top_k: int = 8
    embedding_model: str = "intfloat/e5-large-v2"
    embedding_dim: int = 1024
    eviction_policy: str = "lru"


@dataclass
class GroundingConfig:
    """Grounded-use verification configuration (Equations 6 and 19)."""

    verifier_model: str = "cross-encoder/nli-deberta-v3-large"
    support_threshold: float = 0.8
    verifier_mode: str = "nli"  # Options: "nli", "token_overlap", "citation_only", "retrieve_gated"
    token_overlap_threshold: float = 0.35
    batch_size: int = 32
    max_length: int = 512
    device: str = "auto"


@dataclass
class GroupingConfig:
    """Item-use grouping key and frozen pre-action map configuration (Equation 8)."""

    num_item_buckets: int = 8
    num_context_quartiles: int = 4
    num_difficulty_quintiles: int = 5
    delay_bin_edges: Tuple[int, ...] = (1, 2, 4, 8)  # 5 logarithmic bins: <1, [1,2), [2,4), [4,8), >=8
    min_group_size: int = 2
    max_group_size: int = 8
    no_use_sentinel: str = "NO-USE"
    strategy: str = "item_use"  # Options: "item_use", "exact_state", "listwise_only", "replay_only", "same_request", "domain_only", "random_hash"
    include_difficulty_stratum: bool = True
    key_perturbation_rate: float = 0.0


@dataclass
class RewardConfig:
    """Multi-channel reward weights and normalization configuration (Equations 10-12)."""

    alpha_task: float = 0.60
    alpha_use: float = 0.25
    alpha_cost: float = 0.15
    lambda_spam: float = 0.5
    lambda_store: float = 1e-5
    lambda_ret: float = 0.02
    lambda_ctx: float = 1e-5
    channel_wise_normalization: bool = True
    use_grounded_use: bool = True
    use_grounding_check: bool = True
    use_responsibility_routing: bool = True
    use_use_channel: bool = True
    use_cost_channel: bool = True
    eps: float = 1e-8

    def validate(self) -> None:
        """Ensure channel weights are non-negative and sum to unity."""
        weights = (self.alpha_task, self.alpha_use, self.alpha_cost)
        if any(w < 0.0 for w in weights):
            raise ValueError(f"Channel weights must be non-negative, got {weights}.")
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Channel weights must sum to 1.0, got {total:.6f}.")


@dataclass
class ReplayConfig:
    """Delayed-use replay queue configuration (Algorithm 1 and Appendix E)."""

    enabled: bool = True
    use_replay_groups: bool = True
    use_listwise_groups: bool = True
    replay_ratio: float = 1.0  # 1:1 current-to-replay event pairing
    max_replay_age_sessions: int = 32
    importance_ratio_min: float = 0.5
    importance_ratio_max: float = 2.0
    max_queue_size: int = 100000


@dataclass
class OptimizationConfig:
    """Policy backbone and PPO/GRPO optimization parameters (Equations 13-14, Table 5)."""

    policy_model: str = "Qwen/Qwen2.5-7B-Instruct"
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_updates: int = 500
    rollouts_per_input: int = 8
    effective_batch_size: int = 64
    clip_delta: float = 0.2
    kl_beta: float = 0.01
    max_new_tokens_write: int = 256
    max_new_tokens_answer: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 42
    assignment_operator: str = "ugpo"  # Options: "ugpo", "outcome_grpo", "exact_state", "retrieve_gated", "write_local"
    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class UGPOConfig:
    """Top-level configuration aggregating all UGPO subsystems."""

    experiment_name: str = "ugpo_qwen25_7b"
    output_dir: str = "outputs/ugpo_qwen25_7b"
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def validate(self) -> None:
        """Validate cross-component configuration invariants."""
        self.reward.validate()
        if self.grouping.min_group_size < 2:
            raise ValueError("min_group_size must be at least 2 for group-relative advantages.")
        if self.grouping.max_group_size < self.grouping.min_group_size:
            raise ValueError("max_group_size must be >= min_group_size.")
        if self.replay.importance_ratio_min <= 0.0:
            raise ValueError("importance_ratio_min must be strictly positive.")
        if self.replay.importance_ratio_max <= self.replay.importance_ratio_min:
            raise ValueError("importance_ratio_max must exceed importance_ratio_min.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain nested dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UGPOConfig":
        """Construct a UGPOConfig instance from a nested mapping."""
        grouping_raw = dict(raw.get("grouping", {}))
        if "delay_bin_edges" in grouping_raw and isinstance(grouping_raw["delay_bin_edges"], list):
            grouping_raw["delay_bin_edges"] = tuple(grouping_raw["delay_bin_edges"])

        cfg = cls(
            experiment_name=raw.get("experiment_name", "ugpo_qwen25_7b"),
            output_dir=raw.get("output_dir", "outputs/ugpo_qwen25_7b"),
            memory=MemoryConfig(**raw.get("memory", {})),
            grounding=GroundingConfig(**raw.get("grounding", {})),
            grouping=GroupingConfig(**grouping_raw),
            reward=RewardConfig(**raw.get("reward", {})),
            replay=ReplayConfig(**raw.get("replay", {})),
            optimization=OptimizationConfig(**raw.get("optimization", {})),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UGPOConfig":
        """Load configuration from a YAML file on disk."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    def save_yaml(self, path: str | Path) -> None:
        """Persist configuration as YAML."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["grouping"]["delay_bin_edges"] = list(self.grouping.delay_bin_edges)
        with open(out_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
