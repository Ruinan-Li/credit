"""Item-use grouping keys, frozen pre-action maps, and exact-state coverage diagnostics.

Implements:
- Exact-state coverage C_state(B) and exponential decay bound (Equations 3-4, Appendix B).
- Item-use grouping key kappa(e) = (type(e), phi(i), psi(q), Delta(i, q)) (Equation 8).
- Frozen pre-write K-Means source clusters phi(i) and frozen reference-policy
  entropy difficulty quintiles psi(q) (Section 3.3, Appendix E, Table 5).
- Listwise retrieval groups and cross-request replay groups without double counting.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ugpo.config import GroupingConfig
from ugpo.trace.schema import EventType, RolloutTrace, WRUEvent


@dataclass
class ComparisonGroup:
    """Group K of comparable lifecycle events sharing a grouping key."""

    group_id: str
    group_kind: str  # "answer", "listwise", "replay", "exact_state"
    key: Tuple[Any, ...]
    events: List[WRUEvent] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.events)


def compute_exact_state_coverage(traces: Sequence[RolloutTrace]) -> float:
    """Compute usable comparison coverage C_state(B) under exact-store keys (Equation 3).

    C_state(B) = (1 / |B|) * sum_{b in B} I[|{b' in B : h(s_{b'}) = h(s_b)}| > 1]
    where h(s_b) is the final exact memory-store hash of trajectory b.
    """
    if not traces:
        return 0.0

    hashes: List[str] = []
    for trace in traces:
        # Use the latest logged exact store hash in the trajectory.
        store_hash = ""
        for ev in reversed(trace.events):
            if ev.store_hash:
                store_hash = ev.store_hash
                break
        if not store_hash:
            store_hash = f"empty:{trace.trajectory_id}"
        hashes.append(store_hash)

    counts: Dict[str, int] = {}
    for h in hashes:
        counts[h] = counts.get(h, 0) + 1

    non_singleton = sum(1 for h in hashes if counts[h] > 1)
    return float(non_singleton / len(hashes))


def theoretical_exact_state_bound(batch_size: int, num_writes: int, gamma: float) -> float:
    """Upper bound min{1, (|B| - 1) * gamma^n} on E[C_state(B, n)] (Equation 4)."""
    if batch_size <= 1:
        return 0.0
    return float(min(1.0, (batch_size - 1) * (gamma ** num_writes)))


class FrozenGroupingMaps:
    """Pre-optimization frozen feature maps defining kappa(e) in Equation 8.

    All centroids and quantile edges are fitted once prior to the first policy
    update and never refit on sampled memory content or task rewards (Appendix G).
    """

    def __init__(self, config: GroupingConfig) -> None:
        self.config = config
        self.centroids: Optional[np.ndarray] = None  # Shape: (num_item_buckets, dim)
        self.context_edges: List[float] = [2048.0, 8192.0, 32768.0]
        self.difficulty_edges: List[float] = [0.4, 0.8, 1.2, 1.8]
        self._frozen: bool = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def fit_item_buckets(
        self,
        pre_write_embeddings: np.ndarray,
        max_iter: int = 50,
        seed: int = 42,
    ) -> None:
        """Fit K-Means source clusters on pre-write E5 embeddings before training."""
        if self._frozen:
            raise RuntimeError("Grouping maps are already frozen and cannot be refit.")
        if pre_write_embeddings.ndim != 2 or pre_write_embeddings.shape[0] == 0:
            raise ValueError("pre_write_embeddings must be a non-empty 2D array.")

        n_samples, dim = pre_write_embeddings.shape
        k = min(self.config.num_item_buckets, n_samples)
        rng = np.random.default_rng(seed)

        # Deterministic K-Means++ initialization for stable pre-write source clusters.
        centroids = np.empty((k, dim), dtype=np.float32)
        first_idx = int(rng.integers(0, n_samples))
        centroids[0] = pre_write_embeddings[first_idx]
        min_sq_dist = np.sum((pre_write_embeddings - centroids[0]) ** 2, axis=1)

        for c_idx in range(1, k):
            total = float(min_sq_dist.sum())
            if total <= 1e-12:
                chosen = int(rng.integers(0, n_samples))
            else:
                probs = min_sq_dist / total
                chosen = int(rng.choice(n_samples, p=probs))
            centroids[c_idx] = pre_write_embeddings[chosen]
            dist = np.sum((pre_write_embeddings - centroids[c_idx]) ** 2, axis=1)
            min_sq_dist = np.minimum(min_sq_dist, dist)

        for _ in range(max_iter):
            dists = np.sum(
                (pre_write_embeddings[:, None, :] - centroids[None, :, :]) ** 2,
                axis=2,
            )
            labels = np.argmin(dists, axis=1)
            new_centroids = centroids.copy()
            for c_idx in range(k):
                members = pre_write_embeddings[labels == c_idx]
                if len(members) > 0:
                    mean_vec = members.mean(axis=0)
                    norm = float(np.linalg.norm(mean_vec))
                    new_centroids[c_idx] = mean_vec / norm if norm > 1e-12 else mean_vec
            if np.allclose(new_centroids, centroids, atol=1e-5):
                centroids = new_centroids
                break
            centroids = new_centroids

        self.centroids = centroids.astype(np.float32)

    def fit_task_strata(
        self,
        context_lengths: Sequence[int],
        ref_entropies: Sequence[float],
    ) -> None:
        """Fit context-length quartiles and reference-policy entropy quintiles."""
        if self._frozen:
            raise RuntimeError("Grouping maps are already frozen and cannot be refit.")

        if context_lengths:
            q_points = np.linspace(0.0, 1.0, self.config.num_context_quartiles + 1)[1:-1]
            self.context_edges = [float(v) for v in np.quantile(context_lengths, q_points)]

        if ref_entropies:
            d_points = np.linspace(0.0, 1.0, self.config.num_difficulty_quintiles + 1)[1:-1]
            self.difficulty_edges = [float(v) for v in np.quantile(ref_entropies, d_points)]

    def freeze(self) -> None:
        """Lock all pre-action maps prior to the first policy optimization update."""
        self._frozen = True

    def assign_item_bucket(
        self,
        source_domain: str,
        pre_write_embedding: Optional[np.ndarray] = None,
        fallback_text: str = "",
    ) -> Tuple[str, int]:
        """Map pre-write source domain and segment to frozen bucket phi(i)."""
        if pre_write_embedding is not None and self.centroids is not None:
            vec = np.asarray(pre_write_embedding, dtype=np.float32)
            # Cosine similarity with normalized centroids (fallback to L2 if unnormalized).
            sims = self.centroids @ vec
            cluster_id = int(np.argmax(sims))
        else:
            digest = hashlib.blake2b(
                f"{source_domain}:{fallback_text}".encode("utf-8"),
                digest_size=4,
            ).digest()
            cluster_id = int.from_bytes(digest, "little") % self.config.num_item_buckets
        return (source_domain, cluster_id)

    def assign_task_stratum(
        self,
        benchmark: str,
        request_type: str,
        context_length: int,
        ref_entropy: float,
    ) -> Tuple[Any, ...]:
        """Compute psi(q) = (benchmark, request_type, context_quartile, difficulty_quintile)."""
        ctx_quartile = int(np.searchsorted(self.context_edges, float(context_length), side="right"))
        diff_quintile = int(np.searchsorted(self.difficulty_edges, float(ref_entropy), side="right"))
        if self.config.include_difficulty_stratum:
            return (benchmark, request_type, ctx_quartile, diff_quintile)
        return (benchmark, request_type, ctx_quartile)

    def assign_delay_bin(self, delay_steps_or_sessions: Optional[int]) -> str:
        """Coarsen write-to-use delay Delta(i, q) into five logarithmic bins (Table 5)."""
        if delay_steps_or_sessions is None:
            return self.config.no_use_sentinel
        val = max(0, int(delay_steps_or_sessions))
        bin_idx = 0
        for edge in self.config.delay_bin_edges:
            if val >= edge:
                bin_idx += 1
            else:
                break
        return f"delay_bin_{bin_idx}"

    def save(self, directory: str | Path) -> None:
        """Persist frozen maps to disk for exact training reproducibility."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "context_edges": self.context_edges,
            "difficulty_edges": self.difficulty_edges,
            "num_item_buckets": self.config.num_item_buckets,
            "delay_bin_edges": list(self.config.delay_bin_edges),
            "frozen": self._frozen,
        }
        with open(out_dir / "grouping_maps.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        if self.centroids is not None:
            np.save(out_dir / "item_bucket_centroids.npy", self.centroids)

    @classmethod
    def load(cls, directory: str | Path, config: GroupingConfig) -> "FrozenGroupingMaps":
        """Load previously frozen grouping maps from disk."""
        in_dir = Path(directory)
        maps = cls(config)
        with open(in_dir / "grouping_maps.json", "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        maps.context_edges = list(meta["context_edges"])
        maps.difficulty_edges = list(meta["difficulty_edges"])
        centroid_path = in_dir / "item_bucket_centroids.npy"
        if centroid_path.exists():
            maps.centroids = np.load(centroid_path).astype(np.float32)
        if meta.get("frozen", True):
            maps.freeze()
        return maps


class ItemUseGroupBuilder:
    """Constructs comparison groups K from current-batch and replay events.

    Implements GROUPBYITEMUSE in Algorithm 1 (line 10) and Appendix E:
    1. Current-rollout OUTCOME events use same-input rollout grouping G(tau) (Eq. 2).
    2. When request q retrieves >= 2 items, RETRIEVE events enter a listwise group
       containing all items in S_q.
    3. WRITE and CITE events enter replay groups sharing kappa(e) (Eq. 8).
       Unretrieved writes at trace closure use NO-USE sentinels for psi(q) and Delta(i, q).
    4. No event is compared both listwise and through replay in the same update.
    5. Groups are bounded to [min_group_size, max_group_size] (default [2, 8]).
    """

    def __init__(
        self,
        config: GroupingConfig,
        grouping_maps: Optional[FrozenGroupingMaps] = None,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.grouping_maps = grouping_maps or FrozenGroupingMaps(config)
        self._rng = random.Random(seed)

    def compute_item_use_key(self, event: WRUEvent) -> Tuple[Any, ...]:
        """Evaluate Equation 8 key kappa(e) = (type(e), phi(i), psi(q), Delta(i, q))."""
        strategy = self.config.strategy

        if strategy == "exact_state":
            return (event.event_type.value, event.store_hash or event.trajectory_id)

        if strategy == "random_hash":
            # Table 12 RQ10 stress test: random high-specificity hash key.
            digest = hashlib.blake2b(
                f"{event.event_id}:{event.trajectory_id}".encode("utf-8"),
                digest_size=2,
            ).hexdigest()
            return (event.event_type.value, digest)

        bucket = event.item_bucket if event.item_bucket is not None else 0
        phi = (event.source_domain, bucket)

        if strategy == "domain_only":
            return (event.event_type.value, event.source_domain)

        # Determine psi(q) and Delta(i, q); use NO-USE sentinels for unretrieved writes.
        if event.event_type == EventType.WRITE and not event.is_retrieved:
            psi: Any = self.config.no_use_sentinel
            delta = self.config.no_use_sentinel
        else:
            psi = event.task_stratum if event.task_stratum is not None else self.config.no_use_sentinel
            if event.delay_bin is not None:
                delta = event.delay_bin
            else:
                use_sess = event.metadata.get("use_session", event.session)
                write_sess = event.write_session if event.write_session is not None else event.session
                use_step = event.metadata.get("use_step", event.step)
                write_step = event.write_step if event.write_step is not None else event.step
                raw_delay = max(0, (use_sess - write_sess) if use_sess != write_sess else (use_step - write_step))
                delta = self.grouping_maps.assign_delay_bin(raw_delay)
                event.delay_bin = delta

        # Optional field perturbation for RQ10 robustness evaluation (Table 12).
        if self.config.key_perturbation_rate > 0.0 and self._rng.random() < self.config.key_perturbation_rate:
            perturbed_bucket = self._rng.randrange(self.config.num_item_buckets)
            phi = (event.source_domain, perturbed_bucket)

        if strategy == "same_request":
            return (event.event_type.value, event.request_id or self.config.no_use_sentinel)

        return (event.event_type.value, phi, psi, delta)

    def _partition_by_max_size(
        self,
        group_id_prefix: str,
        group_kind: str,
        key: Tuple[Any, ...],
        events: List[WRUEvent],
    ) -> List[ComparisonGroup]:
        """Split large groups into chunks of size within [min_group_size, max_group_size]."""
        max_sz = self.config.max_group_size
        min_sz = self.config.min_group_size
        if len(events) <= max_sz:
            return [
                ComparisonGroup(
                    group_id=f"{group_id_prefix}:0",
                    group_kind=group_kind,
                    key=key,
                    events=list(events),
                )
            ]

        chunks: List[List[WRUEvent]] = []
        for start in range(0, len(events), max_sz):
            chunks.append(events[start : start + max_sz])

        # Rebalance a singleton tail chunk by borrowing one event from the prior chunk.
        if len(chunks) >= 2 and len(chunks[-1]) < min_sz and len(chunks[-2]) > min_sz:
            needed = min_sz - len(chunks[-1])
            moved = chunks[-2][-needed:]
            chunks[-2] = chunks[-2][:-needed]
            chunks[-1] = moved + chunks[-1]

        return [
            ComparisonGroup(
                group_id=f"{group_id_prefix}:{idx}",
                group_kind=group_kind,
                key=key,
                events=chunk,
            )
            for idx, chunk in enumerate(chunks)
        ]

    def build_groups(
        self,
        current_events: Sequence[WRUEvent],
        replay_events: Sequence[WRUEvent] = (),
        use_listwise: bool = True,
        use_replay: bool = True,
    ) -> List[ComparisonGroup]:
        """Construct disjoint comparison groups across current and replay events."""
        groups: List[ComparisonGroup] = []
        assigned_ids: set[str] = set()

        # 1. Answer generation events always use the same-input rollout group G(tau) (Eq. 2).
        answer_buckets: Dict[Tuple[str, Optional[str]], List[WRUEvent]] = {}
        for ev in current_events:
            if ev.event_type == EventType.OUTCOME:
                key = (ev.input_group_id, ev.request_id)
                answer_buckets.setdefault(key, []).append(ev)
                assigned_ids.add(ev.event_id)

        for idx, (key, evs) in enumerate(answer_buckets.items()):
            groups.extend(
                self._partition_by_max_size(
                    group_id_prefix=f"ans_{idx}",
                    group_kind="answer",
                    key=("OUTCOME", key[0], key[1]),
                    events=evs,
                )
            )

        # 2. Listwise retrieval groups: all items in S_q for each request with |S_q| >= 2.
        strategy = self.config.strategy
        enable_listwise = use_listwise and strategy not in ("replay_only", "exact_state")
        if enable_listwise:
            ret_by_traj_req: Dict[Tuple[str, str], List[WRUEvent]] = {}
            for ev in current_events:
                if ev.event_type == EventType.RETRIEVE and ev.request_id is not None:
                    ret_by_traj_req.setdefault((ev.trajectory_id, ev.request_id), []).append(ev)

            for idx, (key, evs) in enumerate(ret_by_traj_req.items()):
                if len(evs) >= self.config.min_group_size:
                    for ev in evs:
                        assigned_ids.add(ev.event_id)
                    groups.extend(
                        self._partition_by_max_size(
                            group_id_prefix=f"listwise_{idx}",
                            group_kind="listwise",
                            key=("LISTWISE", key[0], key[1]),
                            events=evs,
                        )
                    )

        # 3. Replay / item-use groups sharing kappa(e) for remaining unassigned events.
        enable_replay_grouping = use_replay and strategy != "listwise_only"
        key_buckets: Dict[Tuple[Any, ...], List[WRUEvent]] = {}

        if enable_replay_grouping:
            combined: List[WRUEvent] = []
            for ev in current_events:
                if ev.event_id not in assigned_ids:
                    combined.append(ev)
                    assigned_ids.add(ev.event_id)
            for ev in replay_events:
                if ev.event_id not in assigned_ids:
                    combined.append(ev)
                    assigned_ids.add(ev.event_id)

            for ev in combined:
                k = self.compute_item_use_key(ev)
                key_buckets.setdefault(k, []).append(ev)

            group_kind = "exact_state" if strategy == "exact_state" else "replay"
            for idx, (key, evs) in enumerate(key_buckets.items()):
                groups.extend(
                    self._partition_by_max_size(
                        group_id_prefix=f"{group_kind}_{idx}",
                        group_kind=group_kind,
                        key=key,
                        events=evs,
                    )
                )
        else:
            # Listwise-only ablation: unassigned events remain singletons (contribute 0 local advantage).
            for idx, ev in enumerate(current_events):
                if ev.event_id not in assigned_ids:
                    assigned_ids.add(ev.event_id)
                    groups.append(
                        ComparisonGroup(
                            group_id=f"singleton_{idx}",
                            group_kind="singleton",
                            key=(ev.event_type.value, ev.event_id),
                            events=[ev],
                        )
                    )

        return groups

    @staticmethod
    def group_diagnostics(groups: Sequence[ComparisonGroup], min_group_size: int = 2) -> Dict[str, float]:
        """Compute non-singleton coverage and effective group size (Table 7)."""
        total_events = sum(g.size for g in groups)
        if total_events == 0:
            return {
                "non_singleton_coverage": 0.0,
                "effective_group_size": 0.0,
                "num_groups": 0.0,
            }

        non_singleton_events = sum(g.size for g in groups if g.size >= min_group_size)
        usable_groups = [g for g in groups if g.size >= min_group_size]
        eff_size = (
            float(sum(g.size for g in usable_groups) / len(usable_groups))
            if usable_groups
            else 0.0
        )
        return {
            "non_singleton_coverage": float(non_singleton_events / total_events),
            "effective_group_size": eff_size,
            "num_groups": float(len(groups)),
        }
