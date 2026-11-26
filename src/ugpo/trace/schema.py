"""Write-Retrieve-Use (WRU) event schema and rollout trace containers.

Implements Equation 5:
    e = (type, i, t, q, p, l),  type in {WRITE, RETRIEVE, CITE, OUTCOME}
together with service-side provenance metadata, responsible token spans T(e),
and stored behavior-policy log-probabilities for delayed-use replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EventType(str, Enum):
    """Lifecycle stages tracked in a WRU trace (Equation 5)."""

    WRITE = "WRITE"
    RETRIEVE = "RETRIEVE"
    CITE = "CITE"
    OUTCOME = "OUTCOME"


@dataclass
class TokenSpanData:
    """Policy token sequence and responsible span mask T(e) for one event.

    Following Appendix E:
    - WRITE: emitted memory operation and content span.
    - RETRIEVE: query and ranking-decision span.
    - CITE: identifier-bearing citation span.
    - OUTCOME: full answer token span.

    `behavior_logprobs` stores log pi_{b(e)}(a_j | h_j) for the generated
    tokens so delayed-use replay can evaluate importance ratios (Equation 13)
    without regenerating intervening history.
    """

    input_ids: List[int] = field(default_factory=list)
    prompt_length: int = 0
    responsible_mask: List[bool] = field(default_factory=list)
    behavior_logprobs: List[float] = field(default_factory=list)
    ref_logprobs: Optional[List[float]] = None

    @property
    def response_length(self) -> int:
        """Number of generated tokens following the prompt prefix."""
        return max(0, len(self.input_ids) - self.prompt_length)

    @property
    def num_responsible_tokens(self) -> int:
        """Count of policy tokens covered by T(e)."""
        return sum(1 for flag in self.responsible_mask if flag)


@dataclass
class WRUEvent:
    """Single lifecycle event in an instrumented WRU trace (Equation 5)."""

    event_id: str
    event_type: EventType
    item_id: Optional[str]  # Stable identifier i (None for answer-only OUTCOME events)
    step: int  # Step index t
    request_id: Optional[str] = None  # Downstream request q in Q(tau)
    rank: Optional[int] = None  # Optional retrieval rank p
    predecessor_id: Optional[str] = None  # Predecessor identifier l for updates/deletes

    # Lifecycle and provenance metadata kept outside the model prompt context.
    trajectory_id: str = ""
    input_group_id: str = ""
    session: int = 0
    operation: Optional[str] = None  # ADD, UPDATE, DELETE, NOOP for WRITE events
    item_content: str = ""
    item_tokens: int = 0
    cited_span: str = ""
    answer_text: str = ""
    store_hash: str = ""

    # Frozen pre-action grouping metadata (Section 3.3, Equation 8).
    source_domain: str = "default"
    item_bucket: Optional[int] = None  # phi(i)
    task_stratum: Optional[Tuple[Any, ...]] = None  # psi(q)
    write_step: Optional[int] = None
    write_session: Optional[int] = None
    delay_bin: Optional[str] = None  # Delta(i, q)

    # Verification and reward fields populated during citation validation and routing.
    is_retrieved: bool = False
    is_cited: bool = False
    is_grounded: bool = False
    entailment_score: float = 0.0
    is_valid_citation: bool = False
    task_reward_raw: float = 0.0
    is_successful_answer: bool = False

    # Multi-channel rewards (Equations 10-11) and advantages (Equations 9 and 12).
    reward_task: float = 0.0
    reward_use: float = 0.0
    reward_cost: float = 0.0
    adv_task: float = 0.0
    adv_use: float = 0.0
    adv_cost: float = 0.0
    advantage: float = 0.0

    # Policy token span T(e) and behavior log-probabilities.
    token_span: TokenSpanData = field(default_factory=TokenSpanData)
    is_replay: bool = False
    origin_write_event_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[str, Optional[str], int, Optional[str], Optional[int], Optional[str]]:
        """Return canonical 6-tuple (type, i, t, q, p, l) from Equation 5."""
        return (
            self.event_type.value,
            self.item_id,
            self.step,
            self.request_id,
            self.rank,
            self.predecessor_id,
        )


@dataclass
class RolloutTrace:
    """Complete WRU trace recorded from one rollout trajectory tau."""

    trajectory_id: str
    input_group_id: str
    benchmark: str = "hotpotqa"
    events: List[WRUEvent] = field(default_factory=list)
    terminal_reward: float = 0.0
    store_tokens_at_end: int = 0
    total_generated_tokens: int = 0
    serving_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def events_by_type(self, event_type: EventType) -> List[WRUEvent]:
        """Filter recorded events by lifecycle stage."""
        return [ev for ev in self.events if ev.event_type == event_type]
