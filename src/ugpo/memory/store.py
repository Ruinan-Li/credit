"""Persistent external memory store with stable identifiers and lineage chains.

Implements the memory transition M_t = F(M_{t-1}, o_t, v_t) from Section 2
(Equation 1), supporting ADD, UPDATE, DELETE, and NOOP operations with
service-side provenance tracking, token-budget enforcement, and exact-store
state hashing for state-grouping comparisons (Equation 3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class MemoryOperation(str, Enum):
    """Supported memory-control operations (Equation 1)."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


@dataclass
class SourceSpanMetadata:
    """Pre-write source provenance recorded at write time outside model context.

    The source domain and pre-write text segment feed frozen item-bucket
    assignment phi(i) (Section 3.3), while source_doc_id and char offsets
    enable post-hoc controlled provenance evaluation P_q (Appendix H).
    """

    source_domain: str = "default"
    source_doc_id: str = ""
    char_start: int = 0
    char_end: int = 0
    pre_write_segment: str = ""
    source_unit_id: str = ""


@dataclass
class MemoryItem:
    """Single materialized memory entry stored in M_t."""

    item_id: str
    content: str
    token_count: int
    created_step: int
    created_session: int
    last_accessed_step: int
    predecessor_id: Optional[str] = None
    active: bool = True
    item_bucket: Optional[int] = None
    source_meta: SourceSpanMetadata = field(default_factory=SourceSpanMetadata)


@dataclass
class MemoryTransitionResult:
    """Structured outcome of executing one memory transition step."""

    operation: MemoryOperation
    item_id: Optional[str]
    predecessor_id: Optional[str]
    content: str
    token_count: int
    evicted_ids: List[str] = field(default_factory=list)
    store_hash: str = ""


def default_token_counter(text: str) -> int:
    """Estimate token count using whitespace-delimited subword approximation."""
    if not text:
        return 0
    return max(1, len(text.split()))


class MemoryStore:
    """Persistent external memory store M_t with stable item identifiers.

    Each materialized item receives a deterministic, unique identifier `i`.
    Updates record `predecessor_id` (l in Equation 5) so that credit routed
    through grounded use can traverse or attach to the latest write in the
    lineage chain (Appendix E).
    """

    def __init__(
        self,
        max_memory_tokens: int = 8192,
        token_counter: Optional[Callable[[str], int]] = None,
        id_prefix: str = "mem",
        eviction_policy: str = "lru",
    ) -> None:
        self.max_memory_tokens = max_memory_tokens
        self.token_counter = token_counter or default_token_counter
        self.id_prefix = id_prefix
        self.eviction_policy = eviction_policy

        self._items: Dict[str, MemoryItem] = {}
        self._lineage_head: Dict[str, str] = {}
        self._counter: int = 0

    @property
    def active_items(self) -> List[MemoryItem]:
        """Return all currently live memory items in insertion order."""
        return [item for item in self._items.values() if item.active]

    @property
    def all_items(self) -> Dict[str, MemoryItem]:
        """Return all materialized items, including updated or deleted history."""
        return self._items

    @property
    def active_token_count(self) -> int:
        """Total retained tokens across all live items in M_t."""
        return sum(item.token_count for item in self._items.values() if item.active)

    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Look up a materialized item by its stable identifier."""
        return self._items.get(item_id)

    def resolve_latest_live_id(self, item_id: str) -> Optional[str]:
        """Follow update lineage forward to the latest live identifier, if any."""
        current = item_id
        visited = set()
        while current in self._lineage_head and current not in visited:
            visited.add(current)
            current = self._lineage_head[current]
        item = self._items.get(current)
        if item is not None and item.active:
            return current
        return None

    def get_lineage_chain(self, item_id: str) -> List[str]:
        """Return the predecessor chain [oldest, ..., item_id] for an item."""
        chain: List[str] = []
        current: Optional[str] = item_id
        visited = set()
        while current is not None and current in self._items and current not in visited:
            visited.add(current)
            chain.append(current)
            current = self._items[current].predecessor_id
        chain.reverse()
        return chain

    def mark_accessed(self, item_ids: List[str], step: int) -> None:
        """Record retrieval access timestamps for LRU capacity management."""
        for item_id in item_ids:
            item = self._items.get(item_id)
            if item is not None and item.active:
                item.last_accessed_step = step

    def exact_state_hash(self) -> str:
        """Compute exact high-specificity store identifier h(s_b) (Equation 3).

        Encodes the full ordered set of active item identifiers and contents
        so that any divergent free-form write produces a distinct state key.
        """
        hasher = hashlib.sha256()
        for item in sorted(self.active_items, key=lambda m: m.item_id):
            hasher.update(item.item_id.encode("utf-8"))
            hasher.update(b"\x1f")
            hasher.update(item.content.encode("utf-8"))
            hasher.update(b"\x1e")
        return hasher.hexdigest()

    def _allocate_id(self) -> str:
        self._counter += 1
        return f"{self.id_prefix}_{self._counter:06d}"

    def _enforce_budget(self, protected_id: Optional[str] = None) -> List[str]:
        """Evict active items when retained tokens exceed max_memory_tokens."""
        evicted: List[str] = []
        while self.active_token_count > self.max_memory_tokens:
            candidates = [
                item for item in self.active_items if item.item_id != protected_id
            ]
            if not candidates:
                break
            if self.eviction_policy == "fifo":
                victim = min(candidates, key=lambda m: (m.created_step, m.item_id))
            else:
                victim = min(candidates, key=lambda m: (m.last_accessed_step, m.created_step, m.item_id))
            victim.active = False
            evicted.append(victim.item_id)
        return evicted

    def apply(
        self,
        operation: MemoryOperation | str,
        content: str = "",
        step: int = 0,
        session: int = 0,
        target_id: Optional[str] = None,
        source_meta: Optional[SourceSpanMetadata] = None,
        item_bucket: Optional[int] = None,
    ) -> MemoryTransitionResult:
        """Execute M_t = F(M_{t-1}, o_t, v_t) and return transition metadata.

        Args:
            operation: One of ADD, UPDATE, DELETE, or NOOP.
            content: Free-form memory text for ADD or UPDATE operations.
            step: Current interaction step index t.
            session: Current session index for multi-session workloads.
            target_id: Existing item identifier targeted by UPDATE or DELETE.
            source_meta: Pre-write source provenance metadata.
            item_bucket: Frozen pre-write cluster bucket phi(i).
        """
        op = MemoryOperation(operation) if isinstance(operation, str) else operation
        clean_content = content.strip()

        if op == MemoryOperation.NOOP:
            return MemoryTransitionResult(
                operation=op,
                item_id=None,
                predecessor_id=None,
                content="",
                token_count=0,
                store_hash=self.exact_state_hash(),
            )

        if op == MemoryOperation.ADD:
            if not clean_content:
                return MemoryTransitionResult(
                    operation=MemoryOperation.NOOP,
                    item_id=None,
                    predecessor_id=None,
                    content="",
                    token_count=0,
                    store_hash=self.exact_state_hash(),
                )
            new_id = self._allocate_id()
            tok_count = self.token_counter(clean_content)
            item = MemoryItem(
                item_id=new_id,
                content=clean_content,
                token_count=tok_count,
                created_step=step,
                created_session=session,
                last_accessed_step=step,
                predecessor_id=None,
                active=True,
                item_bucket=item_bucket,
                source_meta=source_meta or SourceSpanMetadata(),
            )
            self._items[new_id] = item
            evicted = self._enforce_budget(protected_id=new_id)
            return MemoryTransitionResult(
                operation=op,
                item_id=new_id,
                predecessor_id=None,
                content=clean_content,
                token_count=tok_count,
                evicted_ids=evicted,
                store_hash=self.exact_state_hash(),
            )

        if op == MemoryOperation.UPDATE:
            live_pred_id = (
                self.resolve_latest_live_id(target_id) if target_id is not None else None
            )
            if live_pred_id is None or not clean_content:
                if clean_content:
                    return self.apply(
                        operation=MemoryOperation.ADD,
                        content=clean_content,
                        step=step,
                        session=session,
                        source_meta=source_meta,
                        item_bucket=item_bucket,
                    )
                return MemoryTransitionResult(
                    operation=MemoryOperation.NOOP,
                    item_id=None,
                    predecessor_id=None,
                    content="",
                    token_count=0,
                    store_hash=self.exact_state_hash(),
                )

            pred_item = self._items[live_pred_id]
            pred_item.active = False
            new_id = self._allocate_id()
            tok_count = self.token_counter(clean_content)
            # Updates preserve the source features of the affected lineage (Appendix E).
            inherited_meta = pred_item.source_meta if source_meta is None else source_meta
            inherited_bucket = pred_item.item_bucket if item_bucket is None else item_bucket
            updated_item = MemoryItem(
                item_id=new_id,
                content=clean_content,
                token_count=tok_count,
                created_step=step,
                created_session=session,
                last_accessed_step=step,
                predecessor_id=live_pred_id,
                active=True,
                item_bucket=inherited_bucket,
                source_meta=inherited_meta,
            )
            self._items[new_id] = updated_item
            self._lineage_head[live_pred_id] = new_id
            evicted = self._enforce_budget(protected_id=new_id)
            return MemoryTransitionResult(
                operation=op,
                item_id=new_id,
                predecessor_id=live_pred_id,
                content=clean_content,
                token_count=tok_count,
                evicted_ids=evicted,
                store_hash=self.exact_state_hash(),
            )

        if op == MemoryOperation.DELETE:
            # Assigns a delete only to the latest live predecessor (Appendix E).
            live_pred_id = (
                self.resolve_latest_live_id(target_id) if target_id is not None else None
            )
            if live_pred_id is None:
                return MemoryTransitionResult(
                    operation=MemoryOperation.NOOP,
                    item_id=None,
                    predecessor_id=None,
                    content="",
                    token_count=0,
                    store_hash=self.exact_state_hash(),
                )
            pred_item = self._items[live_pred_id]
            pred_item.active = False
            return MemoryTransitionResult(
                operation=op,
                item_id=live_pred_id,
                predecessor_id=live_pred_id,
                content=pred_item.content,
                token_count=pred_item.token_count,
                store_hash=self.exact_state_hash(),
            )

        raise ValueError(f"Unsupported memory operation: {operation}")

    def snapshot_stats(self) -> Tuple[int, int]:
        """Return (num_active_items, active_token_count)."""
        active = self.active_items
        return len(active), sum(item.token_count for item in active)
