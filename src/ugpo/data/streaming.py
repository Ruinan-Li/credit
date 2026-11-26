"""Unified streaming document, multi-session, and agentic episode schemas.

Supports the evaluation regimes in Section 4.1 and Appendix H:
- Streaming long-document reasoning (HotpotQA, 2WikiMultiHopQA), where
  documents arrive sequentially rather than as a packed context.
- Multi-session conversational memory (LoCoMo, LongMemEval), where later
  requests consume items written many sessions earlier.
- Agentic transfer (ALFWorld, WebShop) with ReAct think-act episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StreamUnit:
    """Single incoming observation unit streamed to the memory agent at step t."""

    unit_id: str
    text: str
    session_id: int = 0
    source_domain: str = "default"
    source_doc_id: str = ""
    char_start: int = 0
    char_end: int = 0
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldSupportUnit:
    """Post-hoc gold evidence unit used exclusively for controlled provenance P_q.

    Never supplied to training, grouping, or reward construction (Appendix H).
    """

    source_doc_id: str
    char_start: int = 0
    char_end: int = 0
    unit_id: str = ""
    text: str = ""


@dataclass
class RequestItem:
    """Downstream evaluation request q in Q(tau)."""

    request_id: str
    question: str
    answer: str
    request_type: str = "qa"
    session_id: int = 0
    context_length: int = 0
    reference_entropy: float = 0.0
    gold_supports: List[GoldSupportUnit] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Complete streaming or multi-session evaluation episode."""

    episode_id: str
    benchmark: str
    stream_units: List[StreamUnit] = field(default_factory=list)
    requests: List[RequestItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_context_words(self) -> int:
        return sum(len(u.text.split()) for u in self.stream_units)
