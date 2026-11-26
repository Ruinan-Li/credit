"""Grounded-use indicator u_{q,i} and frozen entailment verification.

Implements Equation 6 (QA and conversational benchmarks):
    u_{q,i} = I[i in S_q] * I[i in C_q] * I[g(m_i, y_q) = 1]
and Equation 19 (agentic environments ALFWorld and WebShop):
    u^{env}_{t,i} = I[i in S_t] * I[i in C_t] * I[g_env(m_i, a_t) = 1]

Invalid identifiers and citations lacking a localized supporting span are
rejected before entailment scoring (Section 3.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple

from ugpo.config import GroundingConfig
from ugpo.trace.schema import EventType, RolloutTrace, WRUEvent


# Matches bracketed memory item citations such as [mem_000001] or [id: mem_000001].
_CITATION_PATTERN = re.compile(r"\[(?:id:\s*)?([A-Za-z0-9_\-]+)\]")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "he", "in", "is", "it", "its", "of", "on", "or", "she", "that",
        "the", "their", "they", "this", "to", "was", "were", "will", "with",
    }
)


@dataclass
class CitationSpan:
    """Extracted citation linking a stable item identifier to a localized span."""

    item_id: str
    supporting_span: str
    char_start: int
    char_end: int


@dataclass
class GroundedUseDecision:
    """Verification output for one (memory_item, cited_span) pair."""

    item_id: str
    request_id: str
    is_retrieved: bool
    is_cited: bool
    has_localized_span: bool
    entailment_score: float
    is_grounded: bool

    @property
    def u_qi(self) -> int:
        """Binary grounded-use indicator u_{q,i} in {0, 1} (Equation 6)."""
        return 1 if (self.is_retrieved and self.is_cited and self.is_grounded) else 0


class EntailmentVerifier(Protocol):
    """Protocol for frozen premise-hypothesis entailment verifiers g(m_i, span)."""

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Return support probabilities in [0, 1] for each (premise, hypothesis) pair."""


def extract_citations_and_spans(
    response_text: str,
    valid_prefix: Optional[str] = None,
) -> List[CitationSpan]:
    """Parse identifier citations and localize each to its enclosing sentence span.

    Citations that appear without any non-citation text in their enclosing
    sentence receive an empty `supporting_span` and are subsequently rejected
    prior to entailment scoring (Section 3.1).
    """
    if not response_text:
        return []

    spans: List[CitationSpan] = []
    cursor = 0
    for sentence in _SENTENCE_SPLIT_PATTERN.split(response_text):
        sent_start = response_text.find(sentence, cursor)
        if sent_start < 0:
            sent_start = cursor
        cursor = sent_start + len(sentence)

        matches = list(_CITATION_PATTERN.finditer(sentence))
        if not matches:
            continue

        # Remove citation tags to isolate the localized claim hypothesis.
        claim_text = _CITATION_PATTERN.sub("", sentence).strip()
        claim_text = re.sub(r"\s+", " ", claim_text).strip(" ,;:-")

        for match in matches:
            item_id = match.group(1)
            if valid_prefix is not None and not item_id.startswith(valid_prefix):
                continue
            spans.append(
                CitationSpan(
                    item_id=item_id,
                    supporting_span=claim_text,
                    char_start=sent_start + match.start(),
                    char_end=sent_start + match.end(),
                )
            )
    return spans


class DeBERTaNLIVerifier:
    """Frozen DeBERTaV3-large NLI verifier (Table 5).

    Uses `cross-encoder/nli-deberta-v3-large` with memory item m_i as premise
    and the cited span (or linearized executed action) as hypothesis.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.entailment_idx = self._resolve_entailment_index()

    def _resolve_entailment_index(self) -> int:
        id2label = getattr(self.model.config, "id2label", None) or {}
        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                return int(idx)
        # Standard cross-encoder/nli-deberta-v3-large order: [contradiction, entailment, neutral].
        return 1

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Compute P(entailment | premise=m_i, hypothesis=cited_span) for each pair."""
        if not pairs:
            return []

        scores: List[float] = []
        with self._torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                premises = [p for p, _ in batch]
                hypotheses = [h for _, h in batch]
                encoded = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**encoded).logits
                probs = self._torch.softmax(logits, dim=-1)[:, self.entailment_idx]
                scores.extend(float(v) for v in probs.cpu().tolist())
        return scores


class TokenOverlapVerifier:
    """Content-word overlap verifier used in Table 11 ablation analysis.

    Computes hypothesis-side content-token recall weighted with numeric and
    named-token preservation so unsupported claims fail verification without
    requiring GPU NLI inference.
    """

    def _content_tokens(self, text: str) -> Set[str]:
        tokens = _WORD_PATTERN.findall(text.lower())
        filtered = {t for t in tokens if t not in _STOPWORDS and len(t) > 1}
        return filtered if filtered else set(tokens)

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        scores: List[float] = []
        for premise, hypothesis in pairs:
            h_tokens = self._content_tokens(hypothesis)
            p_tokens = self._content_tokens(premise)
            if not h_tokens or not p_tokens:
                scores.append(0.0)
                continue
            overlap = len(h_tokens & p_tokens) / len(h_tokens)
            # Numeric facts in the hypothesis must appear in the premise to avoid false support.
            h_numbers = {t for t in h_tokens if any(ch.isdigit() for ch in t)}
            if h_numbers and not h_numbers.issubset(p_tokens):
                overlap *= 0.25
            scores.append(float(overlap))
        return scores


class GroundedUseValidator:
    """Applies Equation 6 / Equation 19 across a RolloutTrace (VALIDATECITATIONS).

    Implements Algorithm 1 line 4: validates every recorded citation against
    the retrieved set S_q, localized span requirement, and frozen verifier g.
    Also supports the ablations and controls reported in Tables 3 and 11:
    - Full UGPO: verifier_mode="nli", threshold=0.8
    - Token-overlap verifier: verifier_mode="token_overlap"
    - Citation-only (w/o grounding check): verifier_mode="citation_only"
    - Retrieve-gated (w/o grounded use): verifier_mode="retrieve_gated"
    """

    def __init__(
        self,
        config: GroundingConfig,
        verifier: Optional[EntailmentVerifier] = None,
    ) -> None:
        self.config = config
        if verifier is not None:
            self.verifier: Optional[EntailmentVerifier] = verifier
        elif config.verifier_mode == "nli":
            self.verifier = DeBERTaNLIVerifier(
                model_name=config.verifier_model,
                device=config.device,
                batch_size=config.batch_size,
                max_length=config.max_length,
            )
        elif config.verifier_mode == "token_overlap":
            self.verifier = TokenOverlapVerifier()
        else:
            self.verifier = None

    @property
    def threshold(self) -> float:
        if self.config.verifier_mode == "token_overlap":
            return self.config.token_overlap_threshold
        return self.config.support_threshold

    def validate_trace(
        self,
        trace: RolloutTrace,
        agentic_mode: bool = False,
    ) -> Dict[Tuple[str, str], GroundedUseDecision]:
        """Annotate all CITE, RETRIEVE, and WRITE events in `trace` in place.

        Args:
            trace: Rollout trace containing logged WRU events.
            agentic_mode: When True, uses Equation 19 where the hypothesis is
                the linearized executed action stored in `cited_span`.

        Returns:
            Mapping from (request_id, item_id) to GroundedUseDecision.
        """
        retrieved_by_req: Dict[str, Dict[str, WRUEvent]] = {}
        write_by_item: Dict[str, WRUEvent] = {}
        cite_events: List[WRUEvent] = []

        for ev in trace.events:
            if ev.event_type == EventType.WRITE and ev.item_id is not None:
                write_by_item[ev.item_id] = ev
            elif ev.event_type == EventType.RETRIEVE and ev.request_id is not None and ev.item_id is not None:
                retrieved_by_req.setdefault(ev.request_id, {})[ev.item_id] = ev
            elif ev.event_type == EventType.CITE:
                cite_events.append(ev)

        decisions: Dict[Tuple[str, str], GroundedUseDecision] = {}

        # Retrieve-gated control: treats every retrieved item as used without citation or NLI.
        if self.config.verifier_mode == "retrieve_gated":
            for req_id, ret_map in retrieved_by_req.items():
                for item_id, ret_ev in ret_map.items():
                    ret_ev.is_retrieved = True
                    ret_ev.is_cited = True
                    ret_ev.is_grounded = True
                    ret_ev.is_valid_citation = True
                    ret_ev.entailment_score = 1.0
                    decisions[(req_id, item_id)] = GroundedUseDecision(
                        item_id=item_id,
                        request_id=req_id,
                        is_retrieved=True,
                        is_cited=True,
                        has_localized_span=True,
                        entailment_score=1.0,
                        is_grounded=True,
                    )
            for cite_ev in cite_events:
                req_id = cite_ev.request_id or ""
                item_id = cite_ev.item_id or ""
                in_sq = item_id in retrieved_by_req.get(req_id, {})
                cite_ev.is_retrieved = in_sq
                cite_ev.is_cited = True
                cite_ev.is_grounded = in_sq
                cite_ev.is_valid_citation = in_sq
                cite_ev.entailment_score = 1.0 if in_sq else 0.0
            self._propagate_grounding_to_writes(trace, write_by_item, retrieved_by_req, decisions)
            return decisions

        # Step 1: Reject invalid identifiers and citations lacking a localized span before scoring.
        scoring_events: List[WRUEvent] = []
        scoring_pairs: List[Tuple[str, str]] = []

        for cite_ev in cite_events:
            req_id = cite_ev.request_id or ""
            item_id = cite_ev.item_id or ""
            ret_map = retrieved_by_req.get(req_id, {})
            in_sq = item_id in ret_map
            cite_ev.is_retrieved = in_sq
            cite_ev.is_cited = True

            span_text = cite_ev.cited_span.strip()
            has_span = len(span_text) > 0
            premise = cite_ev.item_content.strip()
            if not premise and in_sq:
                premise = ret_map[item_id].item_content.strip()
                cite_ev.item_content = premise

            if not in_sq or not has_span or not premise:
                cite_ev.is_valid_citation = False
                cite_ev.is_grounded = False
                cite_ev.entailment_score = 0.0
                decisions[(req_id, item_id)] = GroundedUseDecision(
                    item_id=item_id,
                    request_id=req_id,
                    is_retrieved=in_sq,
                    is_cited=True,
                    has_localized_span=has_span,
                    entailment_score=0.0,
                    is_grounded=False,
                )
                continue

            cite_ev.is_valid_citation = True
            if self.config.verifier_mode == "citation_only":
                cite_ev.entailment_score = 1.0
                cite_ev.is_grounded = True
                decisions[(req_id, item_id)] = GroundedUseDecision(
                    item_id=item_id,
                    request_id=req_id,
                    is_retrieved=True,
                    is_cited=True,
                    has_localized_span=True,
                    entailment_score=1.0,
                    is_grounded=True,
                )
            else:
                scoring_events.append(cite_ev)
                scoring_pairs.append((premise, span_text))

        # Step 2: Run frozen verifier on valid (memory_item, cited_span / action) pairs.
        if scoring_pairs and self.verifier is not None:
            scores = self.verifier.score_pairs(scoring_pairs)
            cutoff = self.threshold
            for cite_ev, score in zip(scoring_events, scores):
                req_id = cite_ev.request_id or ""
                item_id = cite_ev.item_id or ""
                grounded = score >= cutoff
                cite_ev.entailment_score = float(score)
                cite_ev.is_grounded = grounded

                key = (req_id, item_id)
                prev = decisions.get(key)
                # If an item is cited multiple times in one request, keep the strongest grounded check.
                if prev is None or (grounded and not prev.is_grounded) or score > prev.entailment_score:
                    decisions[key] = GroundedUseDecision(
                        item_id=item_id,
                        request_id=req_id,
                        is_retrieved=True,
                        is_cited=True,
                        has_localized_span=True,
                        entailment_score=float(score),
                        is_grounded=grounded,
                    )

        # Mirror citation/grounding indicators onto RETRIEVE events and originating WRITE events.
        for req_id, ret_map in retrieved_by_req.items():
            for item_id, ret_ev in ret_map.items():
                dec = decisions.get((req_id, item_id))
                if dec is not None:
                    ret_ev.is_cited = dec.is_cited
                    ret_ev.is_grounded = dec.is_grounded
                    ret_ev.entailment_score = dec.entailment_score
                    ret_ev.is_valid_citation = dec.has_localized_span and dec.is_retrieved

        self._propagate_grounding_to_writes(trace, write_by_item, retrieved_by_req, decisions)
        return decisions

    @staticmethod
    def _propagate_grounding_to_writes(
        trace: RolloutTrace,
        write_by_item: Dict[str, WRUEvent],
        retrieved_by_req: Dict[str, Dict[str, WRUEvent]],
        decisions: Dict[Tuple[str, str], GroundedUseDecision],
    ) -> None:
        """Update z^write and z^use flags on materialized WRITE events."""
        retrieved_items: Set[str] = set()
        for ret_map in retrieved_by_req.values():
            retrieved_items.update(ret_map.keys())

        grounded_items: Set[str] = {
            item_id for (_, item_id), dec in decisions.items() if dec.u_qi == 1
        }
        cited_items: Set[str] = {
            item_id for (_, item_id), dec in decisions.items() if dec.is_cited
        }

        for item_id, write_ev in write_by_item.items():
            write_ev.is_retrieved = item_id in retrieved_items
            write_ev.is_cited = item_id in cited_items
            write_ev.is_grounded = item_id in grounded_items
