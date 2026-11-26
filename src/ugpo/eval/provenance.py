"""Post-hoc controlled provenance alignment P_q and attribution precision/recall.

Implements Appendix H ("Controlled provenance") and Tables 3 & 10:
- Never used as an input to training, grouping, or reward construction.
- Aligns each gold supporting unit of request q to the unique memory identifier
  of the write that first materialized that span using source_doc_id and
  character offsets recorded at write time.
- Computes attribution precision (fraction of positively credited writes in P_q),
  attribution recall (fraction of P_q receiving positive task or use credit),
  and operation-level precision for ADD, UPDATE, and DELETE (Figure 8b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Set

from ugpo.data.streaming import GoldSupportUnit
from ugpo.trace.schema import EventType, RolloutTrace, WRUEvent


@dataclass
class AttributionMetrics:
    """Controlled provenance evaluation metrics (Tables 3, 10, and Figure 8b)."""

    precision_pct: float
    recall_pct: float
    f1_pct: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision_by_operation: Dict[str, float] = field(default_factory=dict)


def _spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Return True if two half-open character intervals intersect (or both are point spans)."""
    if end_a <= start_a or end_b <= start_b:
        return start_a == start_b
    return max(start_a, start_b) < min(end_a, end_b)


def build_reference_provenance(
    trace: RolloutTrace,
    gold_supports: Sequence[GoldSupportUnit],
) -> Set[str]:
    """Align gold supporting units to the first write identifier that materialized each span.

    Returns the reference provenance identifier set P_q defined in Appendix H.
    """
    writes = sorted(
        [ev for ev in trace.events if ev.event_type == EventType.WRITE and ev.item_id is not None],
        key=lambda ev: ev.step,
    )
    reference_ids: Set[str] = set()

    for unit in gold_supports:
        for w_ev in writes:
            w_doc = str(w_ev.metadata.get("source_doc_id", ""))
            w_unit = str(w_ev.metadata.get("source_unit_id", ""))
            w_start = int(w_ev.metadata.get("char_start", 0))
            w_end = int(w_ev.metadata.get("char_end", 0))

            matched = False
            if unit.unit_id and w_unit and unit.unit_id == w_unit:
                matched = True
            elif unit.source_doc_id and w_doc == unit.source_doc_id:
                if _spans_overlap(w_start, w_end, unit.char_start, unit.char_end):
                    matched = True

            if matched and w_ev.item_id is not None:
                # Record the unique memory identifier of the write that first materialized the span.
                reference_ids.add(w_ev.item_id)
                break

    return reference_ids


def evaluate_controlled_provenance(
    traces: Sequence[RolloutTrace],
    gold_supports_by_episode: Mapping[str, Sequence[GoldSupportUnit]],
    credit_threshold: float = 0.0,
) -> AttributionMetrics:
    """Compute attribution precision and recall against post-hoc provenance P_q.

    A write is considered positively credited if its routed task reward, use
    reward, or combined advantage exceeds `credit_threshold`.
    """
    tp = 0
    fp = 0
    fn = 0
    op_tp: Dict[str, int] = {"ADD": 0, "UPDATE": 0, "DELETE": 0}
    op_total_pos: Dict[str, int] = {"ADD": 0, "UPDATE": 0, "DELETE": 0}

    for trace in traces:
        gold_units = gold_supports_by_episode.get(trace.input_group_id, ())
        p_q = build_reference_provenance(trace, gold_units)

        write_events: List[WRUEvent] = [
            ev for ev in trace.events if ev.event_type == EventType.WRITE and ev.item_id is not None
        ]
        credited_ids: Set[str] = set()

        for w_ev in write_events:
            item_id = w_ev.item_id or ""
            is_pos_credited = (
                w_ev.reward_task > credit_threshold
                or w_ev.reward_use > credit_threshold
                or w_ev.advantage > credit_threshold
            )
            if is_pos_credited:
                credited_ids.add(item_id)
                op_name = (w_ev.operation or "ADD").upper()
                if op_name in op_total_pos:
                    op_total_pos[op_name] += 1
                    if item_id in p_q:
                        op_tp[op_name] += 1

        for item_id in credited_ids:
            if item_id in p_q:
                tp += 1
            else:
                fp += 1

        for ref_id in p_q:
            if ref_id not in credited_ids:
                fn += 1

    precision = (100.0 * tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (100.0 * tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    precision_by_op: Dict[str, float] = {}
    for op_name in ("ADD", "UPDATE", "DELETE"):
        denom = op_total_pos[op_name]
        precision_by_op[op_name] = (100.0 * op_tp[op_name] / denom) if denom > 0 else 0.0

    return AttributionMetrics(
        precision_pct=float(precision),
        recall_pct=float(recall),
        f1_pct=float(f1),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision_by_operation=precision_by_op,
    )
