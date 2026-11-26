"""Write-Retrieve-Use (WRU) trace schema, recording, and grounded-use verification."""

from ugpo.trace.grounding import (
    CitationSpan,
    DeBERTaNLIVerifier,
    GroundedUseDecision,
    GroundedUseValidator,
    TokenOverlapVerifier,
    extract_citations_and_spans,
)
from ugpo.trace.recorder import WRUTraceRecorder
from ugpo.trace.schema import EventType, RolloutTrace, TokenSpanData, WRUEvent

__all__ = [
    "CitationSpan",
    "DeBERTaNLIVerifier",
    "EventType",
    "GroundedUseDecision",
    "GroundedUseValidator",
    "RolloutTrace",
    "TokenOverlapVerifier",
    "TokenSpanData",
    "WRUEvent",
    "WRUTraceRecorder",
    "extract_citations_and_spans",
]
