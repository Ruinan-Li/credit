"""Service-side WRU trace recorder with O(E + K) identifier-indexed joins.

Maintains constant-time lookups from stable item identifiers to their
originating write events and predecessor chains so that write-retrieve-use
joins scale linearly in logged events E and retrieved items K (Appendix C).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ugpo.memory.store import MemoryItem, MemoryOperation, MemoryTransitionResult
from ugpo.trace.schema import EventType, RolloutTrace, TokenSpanData, WRUEvent


class WRUTraceRecorder:
    """Records WRITE, RETRIEVE, CITE, and OUTCOME events during a rollout.

    All trace structures live on the memory-service side outside the model's
    context window, matching Section 3.1 and Appendix C.
    """

    def __init__(
        self,
        trajectory_id: str,
        input_group_id: str,
        benchmark: str = "hotpotqa",
    ) -> None:
        self.trajectory_id = trajectory_id
        self.input_group_id = input_group_id
        self.benchmark = benchmark

        self._events: List[WRUEvent] = []
        self._event_counter: int = 0

        # Identifier-indexed maps for O(1) provenance joins.
        self._write_by_item: Dict[str, WRUEvent] = {}
        self._predecessor_map: Dict[str, Optional[str]] = {}
        self._retrieved_by_request: Dict[str, List[WRUEvent]] = {}
        self._retrieved_ids_by_request: Dict[str, Set[str]] = {}
        self._citations_by_request: Dict[str, List[WRUEvent]] = {}
        self._outcome_by_request: Dict[str, WRUEvent] = {}

    def _next_event_id(self, prefix: str) -> str:
        self._event_counter += 1
        return f"{self.trajectory_id}:{prefix}:{self._event_counter:05d}"

    def record_write(
        self,
        transition: MemoryTransitionResult,
        step: int,
        session: int = 0,
        source_domain: str = "default",
        item_bucket: Optional[int] = None,
        token_span: Optional[TokenSpanData] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[WRUEvent]:
        """Record a materialized WRITE (ADD, UPDATE, or DELETE) event."""
        if transition.operation == MemoryOperation.NOOP or transition.item_id is None:
            return None

        event = WRUEvent(
            event_id=self._next_event_id("w"),
            event_type=EventType.WRITE,
            item_id=transition.item_id,
            step=step,
            request_id=None,
            rank=None,
            predecessor_id=transition.predecessor_id,
            trajectory_id=self.trajectory_id,
            input_group_id=self.input_group_id,
            session=session,
            operation=transition.operation.value,
            item_content=transition.content,
            item_tokens=transition.token_count,
            store_hash=transition.store_hash,
            source_domain=source_domain,
            item_bucket=item_bucket,
            write_step=step,
            write_session=session,
            token_span=token_span or TokenSpanData(),
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        self._write_by_item[transition.item_id] = event
        self._predecessor_map[transition.item_id] = transition.predecessor_id
        return event

    def record_retrieval(
        self,
        request_id: str,
        retrieved_items: List[Tuple[MemoryItem, int, float]],
        step: int,
        session: int = 0,
        task_stratum: Optional[Tuple[Any, ...]] = None,
        store_hash: str = "",
        token_span: Optional[TokenSpanData] = None,
    ) -> List[WRUEvent]:
        """Record RETRIEVE events for the ranked candidate set S_q."""
        created: List[WRUEvent] = []
        id_set = self._retrieved_ids_by_request.setdefault(request_id, set())
        req_events = self._retrieved_by_request.setdefault(request_id, [])

        for item, rank, sim_score in retrieved_items:
            origin = self._write_by_item.get(item.item_id)
            event = WRUEvent(
                event_id=self._next_event_id("r"),
                event_type=EventType.RETRIEVE,
                item_id=item.item_id,
                step=step,
                request_id=request_id,
                rank=rank,
                predecessor_id=item.predecessor_id,
                trajectory_id=self.trajectory_id,
                input_group_id=self.input_group_id,
                session=session,
                item_content=item.content,
                item_tokens=item.token_count,
                store_hash=store_hash,
                source_domain=item.source_meta.source_domain,
                item_bucket=item.item_bucket if item.item_bucket is not None else (origin.item_bucket if origin else None),
                task_stratum=task_stratum,
                write_step=item.created_step,
                write_session=item.created_session,
                is_retrieved=True,
                token_span=token_span or TokenSpanData(),
                origin_write_event_id=origin.event_id if origin else None,
                metadata={"retrieval_score": sim_score},
            )
            self._events.append(event)
            req_events.append(event)
            id_set.add(item.item_id)
            created.append(event)

        return created

    def record_citation(
        self,
        request_id: str,
        item_id: str,
        cited_span: str,
        step: int,
        session: int = 0,
        answer_text: str = "",
        task_stratum: Optional[Tuple[Any, ...]] = None,
        store_hash: str = "",
        token_span: Optional[TokenSpanData] = None,
    ) -> WRUEvent:
        """Record a CITE event linking an answer span (or action) to item_id."""
        origin = self.match_origin_write(item_id)
        retrieved_ids = self._retrieved_ids_by_request.get(request_id, set())
        in_retrieved = item_id in retrieved_ids

        # Look up rank from corresponding retrieval event if available.
        rank: Optional[int] = None
        for ret_ev in self._retrieved_by_request.get(request_id, []):
            if ret_ev.item_id == item_id:
                rank = ret_ev.rank
                break

        event = WRUEvent(
            event_id=self._next_event_id("c"),
            event_type=EventType.CITE,
            item_id=item_id,
            step=step,
            request_id=request_id,
            rank=rank,
            predecessor_id=origin.predecessor_id if origin else None,
            trajectory_id=self.trajectory_id,
            input_group_id=self.input_group_id,
            session=session,
            item_content=origin.item_content if origin else "",
            item_tokens=origin.item_tokens if origin else 0,
            cited_span=cited_span,
            answer_text=answer_text,
            store_hash=store_hash,
            source_domain=origin.source_domain if origin else "default",
            item_bucket=origin.item_bucket if origin else None,
            task_stratum=task_stratum,
            write_step=origin.write_step if origin else None,
            write_session=origin.write_session if origin else None,
            is_retrieved=in_retrieved,
            is_cited=True,
            token_span=token_span or TokenSpanData(),
            origin_write_event_id=origin.event_id if origin else None,
        )
        self._events.append(event)
        self._citations_by_request.setdefault(request_id, []).append(event)
        return event

    def record_outcome(
        self,
        request_id: str,
        task_reward: float,
        answer_text: str,
        step: int,
        session: int = 0,
        is_successful: Optional[bool] = None,
        task_stratum: Optional[Tuple[Any, ...]] = None,
        store_hash: str = "",
        token_span: Optional[TokenSpanData] = None,
    ) -> WRUEvent:
        """Record an OUTCOME event carrying terminal request evaluation R_task."""
        success = (task_reward > 0.0) if is_successful is None else is_successful
        event = WRUEvent(
            event_id=self._next_event_id("o"),
            event_type=EventType.OUTCOME,
            item_id=None,
            step=step,
            request_id=request_id,
            rank=None,
            predecessor_id=None,
            trajectory_id=self.trajectory_id,
            input_group_id=self.input_group_id,
            session=session,
            answer_text=answer_text,
            store_hash=store_hash,
            task_stratum=task_stratum,
            task_reward_raw=float(task_reward),
            is_successful_answer=success,
            token_span=token_span or TokenSpanData(),
        )
        self._events.append(event)
        self._outcome_by_request[request_id] = event
        return event

    def match_origin_write(self, item_id: Optional[str]) -> Optional[WRUEvent]:
        """Resolve item_id to the most recent WRITE event in its lineage chain.

        Implements MATCHORIGINWRITE from Algorithm 1 and Appendix E: a grounded
        use joins its stable identifier to the most recent write in its
        predecessor chain so credit follows updates.
        """
        if item_id is None:
            return None
        if item_id in self._write_by_item:
            return self._write_by_item[item_id]
        return None

    def get_retrieved_ids(self, request_id: str) -> Set[str]:
        """Return S_q, the set of retrieved identifiers for request q."""
        return set(self._retrieved_ids_by_request.get(request_id, set()))

    def get_retrievals(self, request_id: str) -> List[WRUEvent]:
        """Return all RETRIEVE events logged for request q."""
        return list(self._retrieved_by_request.get(request_id, []))

    def get_citations(self, request_id: str) -> List[WRUEvent]:
        """Return all CITE events logged for request q."""
        return list(self._citations_by_request.get(request_id, []))

    def get_outcome(self, request_id: str) -> Optional[WRUEvent]:
        """Return the OUTCOME event logged for request q, if present."""
        return self._outcome_by_request.get(request_id)

    def build_trace(
        self,
        terminal_reward: Optional[float] = None,
        store_tokens_at_end: int = 0,
        total_generated_tokens: int = 0,
        serving_latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RolloutTrace:
        """Finalize and return an immutable-style RolloutTrace container."""
        if terminal_reward is None:
            outcomes = [ev.task_reward_raw for ev in self._outcome_by_request.values()]
            terminal_reward = float(sum(outcomes) / len(outcomes)) if outcomes else 0.0

        return RolloutTrace(
            trajectory_id=self.trajectory_id,
            input_group_id=self.input_group_id,
            benchmark=self.benchmark,
            events=list(self._events),
            terminal_reward=float(terminal_reward),
            store_tokens_at_end=store_tokens_at_end,
            total_generated_tokens=total_generated_tokens,
            serving_latency_ms=serving_latency_ms,
            metadata=dict(metadata or {}),
        )
