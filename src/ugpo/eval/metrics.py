"""Task accuracy, lifecycle utilization, pollution, and serving metrics (Tables 1 and 6).

Implements:
- Answer F1 and Exact Match for HotpotQA and 2WikiMultiHopQA.
- Category-aware official scoring for LoCoMo and LongMemEval.
- Four-task unweighted macro-average.
- Lifecycle and serving metrics: write utilization (%), retrieval precision (%),
  unsupported citations (%), memory pollution (tokens), store size (tokens),
  retrieved tokens per request, and serving latency (ms).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Set

from ugpo.trace.schema import EventType, RolloutTrace


_CITATION_TAG_RE = re.compile(r"\[(?:id:\s*)?[A-Za-z0-9_\-]+\]")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_ABSTENTION_PHRASES = (
    "no information available",
    "not mentioned",
    "cannot be determined",
    "unknown",
    "unanswerable",
    "not provided",
    "no evidence",
)


@dataclass
class LifecycleMetrics:
    """Serving-side and memory-lifecycle metrics matching Table 6."""

    write_utilization_pct: float
    retrieval_precision_pct: float
    unsupported_citations_pct: float
    pollution_tokens: float
    store_size_tokens: float
    retrieved_tokens_per_request: float
    serving_latency_ms: float


def strip_citations(text: str) -> str:
    """Remove inline `[mem_XXXXXX]` identifier tags before answer scoring."""
    return _CITATION_TAG_RE.sub("", text).strip()


def normalize_answer(text: str) -> str:
    """Standard SQuAD / HotpotQA normalization after stripping inline citations."""
    cleaned = strip_citations(text).lower()
    cleaned = "".join(ch for ch in cleaned if ch not in string.punctuation)
    cleaned = _ARTICLES_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match score in {0.0, 1.0}."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score in [0.0, 1.0] used by HotpotQA and 2WikiMultiHopQA."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    # Standard HotpotQA yes/no/noanswer strict check.
    if gold_tokens in (["yes"], ["no"], ["noanswer"]) and pred_tokens != gold_tokens:
        return 0.0
    if pred_tokens in (["yes"], ["no"], ["noanswer"]) and pred_tokens != gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return float(2.0 * precision * recall / (precision + recall))


def compute_locomo_score(
    prediction: str,
    ground_truth: str,
    category: int = 1,
) -> float:
    """Official category-aware LoCoMo evaluation score.

    Categories 1-4 (single-hop, temporal, open-domain, multi-hop) use
    normalized token F1 (with multi-answer comma splitting for category 1).
    Category 5 (adversarial unanswerable questions) awards 1.0 when the
    response appropriately abstains and avoids hallucinating the adversarial decoy.
    """
    if category == 5:
        lower_pred = strip_citations(prediction).lower()
        if any(phrase in lower_pred for phrase in _ABSTENTION_PHRASES):
            return 1.0
        return 0.0

    if category == 1 and "," in ground_truth:
        sub_golds = [g.strip() for g in ground_truth.split(",") if g.strip()]
        if sub_golds:
            sub_preds = [p.strip() for p in strip_citations(prediction).split(",") if p.strip()] or [prediction]
            scores = [max(compute_f1(p, g) for p in sub_preds) for g in sub_golds]
            return float(sum(scores) / len(scores))

    return compute_f1(prediction, ground_truth)


def compute_longmemeval_score(prediction: str, ground_truth: str) -> float:
    """Normalized accuracy/F1 evaluation score for LongMemEval."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)
    if not norm_gold:
        return 1.0 if not norm_pred else 0.0
    if norm_gold in norm_pred:
        return 1.0
    return compute_f1(prediction, ground_truth)


def compute_task_reward(
    benchmark: str,
    prediction: str,
    ground_truth: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> float:
    """Dispatch terminal request reward R_task according to benchmark protocol."""
    name = benchmark.lower().strip()
    meta = metadata or {}
    if name in ("hotpotqa", "2wiki", "2wikimultihopqa"):
        return compute_f1(prediction, ground_truth)
    if name == "locomo":
        category = int(meta.get("category", 1))
        return compute_locomo_score(prediction, ground_truth, category=category)
    if name == "longmemeval":
        return compute_longmemeval_score(prediction, ground_truth)
    if name in ("alfworld", "webshop"):
        # Environment success binary indicator.
        return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) and bool(ground_truth) else 0.0
    return compute_f1(prediction, ground_truth)


def compute_four_task_macro_average(benchmark_scores: Mapping[str, float]) -> float:
    """Compute the unweighted macro-average over HotpotQA, 2Wiki, LoCoMo, LongMemEval."""
    required = ("hotpotqa", "2wiki", "locomo", "longmemeval")
    present = [float(benchmark_scores[k]) for k in required if k in benchmark_scores]
    if not present:
        return 0.0
    return float(sum(present) / len(present))


def compute_lifecycle_metrics(traces: Sequence[RolloutTrace]) -> LifecycleMetrics:
    """Aggregate Table 6 lifecycle, pollution, and serving metrics across traces."""
    if not traces:
        return LifecycleMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total_writes = 0
    grounded_writes = 0
    total_retrievals = 0
    grounded_retrievals = 0
    total_citations = 0
    unsupported_citations = 0

    pollution_per_trace = []
    store_sizes = []
    retrieved_tokens_per_req = []
    latencies = []

    for trace in traces:
        # Track unique materialized items written in this trace.
        written_tokens: Dict[str, int] = {}
        grounded_items: Set[str] = set()
        req_ret_tokens: Dict[str, int] = {}

        for ev in trace.events:
            if ev.event_type == EventType.WRITE and ev.item_id is not None:
                if ev.operation != "DELETE":
                    written_tokens[ev.item_id] = ev.item_tokens
                if ev.is_grounded:
                    grounded_items.add(ev.item_id)
            elif ev.event_type == EventType.RETRIEVE:
                total_retrievals += 1
                if ev.is_cited and ev.is_grounded:
                    grounded_retrievals += 1
                    if ev.item_id is not None:
                        grounded_items.add(ev.item_id)
                req_key = ev.request_id or "_default"
                req_ret_tokens[req_key] = req_ret_tokens.get(req_key, 0) + ev.item_tokens
            elif ev.event_type == EventType.CITE:
                total_citations += 1
                if ev.is_grounded and ev.is_retrieved:
                    if ev.item_id is not None:
                        grounded_items.add(ev.item_id)
                else:
                    unsupported_citations += 1

        total_writes += len(written_tokens)
        grounded_writes += sum(1 for item_id in written_tokens if item_id in grounded_items)
        trace_pollution = sum(
            toks for item_id, toks in written_tokens.items() if item_id not in grounded_items
        )
        pollution_per_trace.append(float(trace_pollution))
        store_sizes.append(float(trace.store_tokens_at_end or sum(written_tokens.values())))
        retrieved_tokens_per_req.extend(float(v) for v in req_ret_tokens.values())
        latencies.append(float(trace.serving_latency_ms))

    write_util = (100.0 * grounded_writes / total_writes) if total_writes > 0 else 0.0
    ret_prec = (100.0 * grounded_retrievals / total_retrievals) if total_retrievals > 0 else 0.0
    unsup_pct = (100.0 * unsupported_citations / total_citations) if total_citations > 0 else 0.0

    return LifecycleMetrics(
        write_utilization_pct=float(write_util),
        retrieval_precision_pct=float(ret_prec),
        unsupported_citations_pct=float(unsup_pct),
        pollution_tokens=float(sum(pollution_per_trace) / len(pollution_per_trace)),
        store_size_tokens=float(sum(store_sizes) / len(store_sizes)),
        retrieved_tokens_per_request=(
            float(sum(retrieved_tokens_per_req) / len(retrieved_tokens_per_req))
            if retrieved_tokens_per_req
            else 0.0
        ),
        serving_latency_ms=float(sum(latencies) / len(latencies)),
    )
