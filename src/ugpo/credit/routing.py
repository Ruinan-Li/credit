"""Responsibility masks and multi-channel reward routing (Equations 7, 10, 11).

Routes task outcome, grounded use, and memory cost signals exclusively to the
operations that participate in each item's observed Write-Retrieve-Use
lifecycle, preventing trajectory-wide credit dilution and protecting grounded
writes when downstream answer generation fails (Section 3.2, Section 3.4,
Table 15).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from ugpo.config import RewardConfig
from ugpo.trace.schema import EventType, RolloutTrace, WRUEvent


@dataclass
class ResponsibilityMasks:
    """Binary eligibility masks defined in Equation 7."""

    z_write: Dict[str, int] = field(default_factory=dict)  # item_id -> I[sum_q I[i in S_q] > 0]
    z_use: Dict[str, int] = field(default_factory=dict)  # item_id -> I[sum_q u_{q,i} > 0]
    z_ret: Dict[Tuple[str, str], int] = field(default_factory=dict)  # (q, i) -> I[i in S_q]
    z_ans: Dict[str, int] = field(default_factory=dict)  # q -> 1


class ResponsibilityRouter:
    """Computes Equation 7 masks and assigns task, use, and cost channel rewards.

    Implements ROUTEREWARDS in Algorithm 1 (line 5).
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def compute_masks(self, trace: RolloutTrace) -> ResponsibilityMasks:
        """Compute z^write, z^use, z^ret, and z^ans over all events in trace."""
        masks = ResponsibilityMasks()
        retrieved_items: Set[str] = set()
        used_items: Set[str] = set()

        for ev in trace.events:
            if ev.event_type == EventType.WRITE and ev.item_id is not None:
                masks.z_write.setdefault(ev.item_id, 0)
                masks.z_use.setdefault(ev.item_id, 0)
            elif ev.event_type == EventType.RETRIEVE and ev.request_id is not None and ev.item_id is not None:
                retrieved_items.add(ev.item_id)
                masks.z_ret[(ev.request_id, ev.item_id)] = 1
                if self._event_is_used(ev):
                    used_items.add(ev.item_id)
            elif ev.event_type == EventType.CITE and ev.item_id is not None:
                if self._event_is_used(ev):
                    used_items.add(ev.item_id)
            elif ev.event_type == EventType.OUTCOME and ev.request_id is not None:
                masks.z_ans[ev.request_id] = 1

        for item_id in retrieved_items:
            masks.z_write[item_id] = 1
        for item_id in used_items:
            masks.z_use[item_id] = 1

        return masks

    def _event_is_used(self, ev: WRUEvent) -> bool:
        """Evaluate u_{q,i} according to active ablation settings."""
        if not self.config.use_grounded_use:
            return ev.is_retrieved
        if not self.config.use_grounding_check:
            return ev.is_retrieved and ev.is_cited and ev.is_valid_citation
        return ev.is_retrieved and ev.is_cited and ev.is_grounded

    def route_trace(self, trace: RolloutTrace) -> ResponsibilityMasks:
        """Populate reward_task, reward_use, and reward_cost on every event in trace.

        Follows Equations 7, 10, 11 and Table 15:
        - Unretrieved write: task=0, use=0, cost=-lambda_store * tok(m_i).
        - Retrieved but unused item: retrieval receives cost and no task credit.
        - Grounded item in unsuccessful answer: upstream write task=0 while
          answer retains terminal outcome; use channel still credits grounding.
        - Grounded item in successful answer: positive task credit routed to
          the responsible write (latest in predecessor chain) and retrieval.
        - Unsupported citation: use=-lambda_spam on the citation span; no task
          credit path opened to the originating write.
        """
        masks = self.compute_masks(trace)

        outcome_by_req: Dict[str, WRUEvent] = {}
        write_by_item: Dict[str, WRUEvent] = {}
        grounded_requests_by_item: Dict[str, List[WRUEvent]] = {}
        retrieved_requests_by_item: Dict[str, List[WRUEvent]] = {}

        for ev in trace.events:
            if ev.event_type == EventType.OUTCOME and ev.request_id is not None:
                outcome_by_req[ev.request_id] = ev
            elif ev.event_type == EventType.WRITE and ev.item_id is not None:
                write_by_item[ev.item_id] = ev

        # Collect downstream request outcomes linked to each retrieved/grounded item.
        for ev in trace.events:
            if ev.event_type == EventType.RETRIEVE and ev.request_id is not None and ev.item_id is not None:
                out_ev = outcome_by_req.get(ev.request_id)
                if out_ev is not None:
                    retrieved_requests_by_item.setdefault(ev.item_id, []).append(out_ev)
                    if self._event_is_used(ev):
                        grounded_requests_by_item.setdefault(ev.item_id, []).append(out_ev)

        # Assign channel rewards to every lifecycle event.
        for ev in trace.events:
            ev.reward_task = 0.0
            ev.reward_use = 0.0
            ev.reward_cost = 0.0

            if ev.event_type == EventType.OUTCOME:
                # Answer generation always retains the terminal request outcome (z^ans_q = 1).
                ev.reward_task = float(ev.task_reward_raw)
                ev.reward_use = 0.0
                ev.reward_cost = 0.0
                continue

            if ev.event_type == EventType.CITE:
                used = self._event_is_used(ev)
                out_ev = outcome_by_req.get(ev.request_id or "")

                if not self.config.use_responsibility_routing:
                    ev.reward_task = float(out_ev.task_reward_raw if out_ev else trace.terminal_reward)
                elif used and out_ev is not None and out_ev.is_successful_answer and out_ev.task_reward_raw > 0.0:
                    ev.reward_task = float(out_ev.task_reward_raw)
                else:
                    # Unsuccessful answer or unsupported citation does not open positive task credit.
                    ev.reward_task = 0.0

                if self.config.use_use_channel:
                    # Equation 10: u_{q,i} - lambda_spam * I[i in C_q] * I[g(m_i, y_q) = 0].
                    if used:
                        ev.reward_use = 1.0
                    elif ev.is_cited:
                        ev.reward_use = -float(self.config.lambda_spam)
                ev.reward_cost = 0.0
                continue

            if ev.event_type == EventType.RETRIEVE:
                used = self._event_is_used(ev)
                out_ev = outcome_by_req.get(ev.request_id or "")

                if not self.config.use_responsibility_routing:
                    ev.reward_task = float(out_ev.task_reward_raw if out_ev else trace.terminal_reward)
                elif used and out_ev is not None and out_ev.is_successful_answer and out_ev.task_reward_raw > 0.0:
                    ev.reward_task = float(out_ev.task_reward_raw)
                else:
                    ev.reward_task = 0.0

                if self.config.use_use_channel:
                    if used:
                        ev.reward_use = 1.0
                    elif ev.is_cited and not ev.is_grounded and self.config.use_grounding_check:
                        ev.reward_use = -float(self.config.lambda_spam)
                    else:
                        ev.reward_use = 0.0

                if self.config.use_cost_channel:
                    # Equation 11: -lambda_ret - lambda_ctx * tok(m_i).
                    ev.reward_cost = -float(self.config.lambda_ret) - float(self.config.lambda_ctx) * float(ev.item_tokens)
                continue

            if ev.event_type == EventType.WRITE:
                item_id = ev.item_id or ""
                used_flag = masks.z_use.get(item_id, 0) > 0
                grounded_outcomes = grounded_requests_by_item.get(item_id, [])

                if not self.config.use_responsibility_routing:
                    ev.reward_task = float(trace.terminal_reward)
                elif used_flag and grounded_outcomes:
                    # Successful grounded use routes positive task credit to originating write;
                    # failed downstream answers contribute 0 rather than negative task penalty (Section 3.4, Table 15).
                    positive_scores = [
                        float(o.task_reward_raw)
                        for o in grounded_outcomes
                        if o.is_successful_answer and o.task_reward_raw > 0.0
                    ]
                    ev.reward_task = (sum(positive_scores) / len(positive_scores)) if positive_scores else 0.0
                else:
                    ev.reward_task = 0.0

                if self.config.use_use_channel:
                    # Grounded consumption rewards the originating write; unsupported citations
                    # are charged on the citation event and do not penalize a live write (Table 15).
                    ev.reward_use = 1.0 if used_flag else 0.0

                if self.config.use_cost_channel:
                    # Equation 11: -lambda_store * tok(m_i) for write events.
                    ev.reward_cost = -float(self.config.lambda_store) * float(ev.item_tokens)

                # Attach downstream request stratum and write-to-use delay metadata when grounded.
                if grounded_outcomes:
                    first_out = grounded_outcomes[0]
                    ev.task_stratum = first_out.task_stratum
                    ev.request_id = first_out.request_id
                    ev.metadata["use_step"] = first_out.step
                    ev.metadata["use_session"] = first_out.session

        return masks
