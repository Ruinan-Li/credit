"""Delayed-use replay queue Q_delay for cross-session credit propagation.

Implements Section 3.4, Algorithm 1 (lines 6-9), and Appendix E:
- Joins future grounded-use events with originating writes via (i, t_write, t_use).
- Reuses stored behavior-policy log-probabilities log pi_{b(e)}(a_j | h_j) without
  reproducing intervening interaction history.
- Enforces 32-session expiry and pre-clipping importance-ratio filtering [0.5, 2.0].
- Pairs current optimization batch events 1:1 with uniformly sampled replay entries
  matching the same grouping key kappa(e).
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from ugpo.config import ReplayConfig
from ugpo.trace.schema import EventType, WRUEvent


@dataclass
class DelayedUseReplayEntry:
    """Single write-use provenance pair stored in Q_delay."""

    write_event: WRUEvent
    use_event: Optional[WRUEvent]
    grouping_key: Tuple[Any, ...]
    enqueued_session: int
    write_step: int
    use_step: int


class DelayedUseReplayQueue:
    """Bounded replay buffer keyed by frozen item-use key kappa(e).

    Maintains both:
    1. An index of historical write events (keyed by item_id) so that when a
       later session grounds item `i`, `match_origin_write` retrieves the
       originating write and its stored behavior log-probabilities in O(1).
    2. Key-partitioned replay queues supporting 1:1 uniform sampling within
       matching kappa(e) subject to session-age and importance-ratio bounds.
    """

    def __init__(self, config: ReplayConfig, seed: int = 42) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self._write_archive: Dict[str, WRUEvent] = {}
        self._predecessor_archive: Dict[str, Optional[str]] = {}
        self._entries_by_key: Dict[Tuple[Any, ...], Deque[DelayedUseReplayEntry]] = {}
        self._total_entries: int = 0

    def __len__(self) -> int:
        return self._total_entries

    def register_write(self, write_event: WRUEvent) -> None:
        """Archive a materialized WRITE event for future cross-session joins."""
        if write_event.event_type != EventType.WRITE or write_event.item_id is None:
            return
        self._write_archive[write_event.item_id] = copy.deepcopy(write_event)
        self._predecessor_archive[write_event.item_id] = write_event.predecessor_id

    def match_origin_write(self, item_id: Optional[str]) -> Optional[WRUEvent]:
        """Resolve a delayed-use item identifier to its originating WRITE event."""
        if item_id is None:
            return None
        return self._write_archive.get(item_id)

    def enqueue_delayed_pair(
        self,
        origin_write: WRUEvent,
        use_event: Optional[WRUEvent],
        grouping_key: Tuple[Any, ...],
        current_session: int,
    ) -> DelayedUseReplayEntry:
        """Join delayed use with origin_write and insert into Q_delay."""
        replayed_write = copy.deepcopy(origin_write)
        replayed_write.is_replay = True
        replayed_write.event_id = f"{origin_write.event_id}:replay:{current_session}:{self._total_entries}"

        if use_event is not None:
            replayed_write.is_retrieved = True
            replayed_write.is_cited = True
            replayed_write.is_grounded = use_event.is_grounded
            replayed_write.request_id = use_event.request_id
            replayed_write.task_stratum = use_event.task_stratum
            replayed_write.delay_bin = use_event.delay_bin
            replayed_write.reward_task = use_event.reward_task
            replayed_write.reward_use = max(replayed_write.reward_use, use_event.reward_use)
            replayed_write.metadata["use_step"] = use_event.step
            replayed_write.metadata["use_session"] = use_event.session
            use_step = use_event.step
        else:
            use_step = origin_write.step

        entry = DelayedUseReplayEntry(
            write_event=replayed_write,
            use_event=copy.deepcopy(use_event) if use_event is not None else None,
            grouping_key=grouping_key,
            enqueued_session=current_session,
            write_step=origin_write.write_step if origin_write.write_step is not None else origin_write.step,
            use_step=use_step,
        )

        bucket = self._entries_by_key.setdefault(grouping_key, deque())
        bucket.append(entry)
        self._total_entries += 1
        self._enforce_capacity()
        return entry

    def _enforce_capacity(self) -> None:
        while self._total_entries > self.config.max_queue_size:
            # Evict oldest entry from the largest bucket.
            largest_key = max(self._entries_by_key, key=lambda k: len(self._entries_by_key[k]))
            if self._entries_by_key[largest_key]:
                self._entries_by_key[largest_key].popleft()
                self._total_entries -= 1
                if not self._entries_by_key[largest_key]:
                    del self._entries_by_key[largest_key]
            else:
                break

    def prune_expired(self, current_session: int) -> int:
        """Remove replay entries older than max_replay_age_sessions (default 32)."""
        removed = 0
        max_age = self.config.max_replay_age_sessions
        empty_keys: List[Tuple[Any, ...]] = []

        for key, bucket in self._entries_by_key.items():
            kept: Deque[DelayedUseReplayEntry] = deque()
            for entry in bucket:
                age = current_session - entry.enqueued_session
                if age <= max_age:
                    kept.append(entry)
                else:
                    removed += 1
            if kept:
                self._entries_by_key[key] = kept
            else:
                empty_keys.append(key)

        for key in empty_keys:
            del self._entries_by_key[key]

        self._total_entries -= removed
        return removed

    def is_ratio_within_bounds(
        self,
        current_logprobs: Sequence[float],
        behavior_logprobs: Sequence[float],
        responsible_mask: Sequence[bool],
    ) -> bool:
        """Check whether every responsible-token ratio lies in [0.5, 2.0] (Appendix E).

        An entry is discarded before objective clipping whenever any
        responsible-token behavior ratio rho_j(theta) falls outside
        [importance_ratio_min, importance_ratio_max].
        """
        lo = self.config.importance_ratio_min
        hi = self.config.importance_ratio_max

        for cur_lp, beh_lp, active in zip(current_logprobs, behavior_logprobs, responsible_mask):
            if not active:
                continue
            log_ratio = float(cur_lp) - float(beh_lp)
            # Guard against numeric overflow before exponentiation.
            if log_ratio > 20.0 or log_ratio < -20.0:
                return False
            ratio = math.exp(log_ratio)
            if ratio < lo or ratio > hi:
                return False
        return True

    def sample_matching_replay(
        self,
        current_events: Sequence[WRUEvent],
        key_fn: Callable[[WRUEvent], Tuple[Any, ...]],
        current_session: int,
        logprob_evaluator: Optional[Callable[[WRUEvent], Sequence[float]]] = None,
    ) -> Tuple[List[WRUEvent], Dict[str, int]]:
        """Sample replay events matching current_events keys at 1:1 ratio.

        Args:
            current_events: Current optimization batch events.
            key_fn: Function mapping an event to its frozen key kappa(e).
            current_session: Current session counter for age expiry.
            logprob_evaluator: Optional callable returning current policy token
                log-probabilities for ratio truncation filtering [0.5, 2.0].

        Returns:
            (sampled_replay_events, diagnostics) where diagnostics counts
            expired and ratio-truncated entries.
        """
        if not self.config.enabled or not self.config.use_replay_groups:
            return [], {"expired": 0, "ratio_discarded": 0, "sampled": 0}

        expired = self.prune_expired(current_session)
        ratio_discarded = 0
        sampled: List[WRUEvent] = []

        # Count how many current events fall into each grouping key.
        counts_by_key: Dict[Tuple[Any, ...], int] = {}
        for ev in current_events:
            if ev.event_type in (EventType.WRITE, EventType.CITE):
                k = key_fn(ev)
                counts_by_key[k] = counts_by_key.get(k, 0) + 1

        for key, cur_count in counts_by_key.items():
            bucket = self._entries_by_key.get(key)
            if not bucket:
                continue

            target = max(1, int(round(cur_count * self.config.replay_ratio)))
            candidates = list(bucket)
            self._rng.shuffle(candidates)

            chosen_for_key = 0
            valid_survivors: Deque[DelayedUseReplayEntry] = deque()
            discarded_ids = set()

            for entry in candidates:
                ev = entry.write_event
                if logprob_evaluator is not None and ev.token_span.behavior_logprobs:
                    cur_lps = logprob_evaluator(ev)
                    if not self.is_ratio_within_bounds(
                        current_logprobs=cur_lps,
                        behavior_logprobs=ev.token_span.behavior_logprobs,
                        responsible_mask=ev.token_span.responsible_mask,
                    ):
                        ratio_discarded += 1
                        discarded_ids.add(ev.event_id)
                        continue
                if chosen_for_key < target:
                    sampled.append(ev)
                    chosen_for_key += 1

            # Drop entries whose policy drift exceeded the [0.5, 2.0] ratio cutoff.
            if discarded_ids:
                for entry in bucket:
                    if entry.write_event.event_id not in discarded_ids:
                        valid_survivors.append(entry)
                if valid_survivors:
                    self._entries_by_key[key] = valid_survivors
                else:
                    del self._entries_by_key[key]
                self._total_entries -= len(discarded_ids)

        return sampled, {
            "expired": expired,
            "ratio_discarded": ratio_discarded,
            "sampled": len(sampled),
        }
