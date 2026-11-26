"""UGPO Training Procedure (Algorithm 1) and matched credit-assignment baselines.

Orchestrates the full optimization step:
  1. Rollout memory agent pi_theta with persistent MemoryStore and WRUTraceRecorder.
  2. Validate citations via GroundedUseValidator (Eq. 6 / Eq. 19).
  3. Route task, use, and cost rewards via ResponsibilityRouter (Eq. 7, 10, 11).
  4. Match delayed-use events to originating writes and insert into Q_delay.
  5. Construct comparison groups (listwise + replay + answer) via ItemUseGroupBuilder (Eq. 8).
  6. Normalize channels independently and combine advantages (Eq. 9 & 12).
  7. Optimize L_UGPO(theta) with clipped importance ratios and KL regularization (Eq. 13-14).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ugpo.config import UGPOConfig
from ugpo.credit.grouping import (
    ComparisonGroup,
    FrozenGroupingMaps,
    ItemUseGroupBuilder,
    compute_exact_state_coverage,
)
from ugpo.credit.normalization import (
    ChannelNormalizationStats,
    normalize_and_combine_advantages,
)
from ugpo.credit.replay import DelayedUseReplayQueue
from ugpo.credit.routing import ResponsibilityRouter
from ugpo.data.streaming import Episode
from ugpo.memory.retriever import DenseMemoryRetriever, TextEncoder
from ugpo.memory.store import MemoryStore, SourceSpanMetadata
from ugpo.policy.agent import MemoryPolicyAgent, build_token_span_mask
from ugpo.policy.loss import UGPOLossOutput, compute_sequence_logprobs, compute_ugpo_loss
from ugpo.trace.grounding import GroundedUseValidator, extract_citations_and_spans
from ugpo.trace.recorder import WRUTraceRecorder
from ugpo.trace.schema import EventType, RolloutTrace, TokenSpanData, WRUEvent


@dataclass
class TrainingStepMetrics:
    """Comprehensive telemetry emitted by one Algorithm 1 optimization step."""

    step: int
    loss: float
    policy_loss: float
    kl_loss: float
    clip_fraction: float
    mean_ratio: float
    mean_task_reward: float
    non_singleton_coverage: float
    exact_state_coverage: float
    effective_group_size: float
    task_channel_sd: float
    use_channel_sd: float
    cost_channel_sd: float
    num_current_events: int
    num_replay_events: int
    replay_expired: int
    replay_ratio_discarded: int
    trace_processing_ms: float
    extra: Dict[str, Any] = field(default_factory=dict)


class UGPOTrainer:
    """End-to-end trainer implementing Algorithm 1 and Table 14 assignment operators."""

    def __init__(
        self,
        config: UGPOConfig,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module],
        tokenizer: Any,
        encoder: TextEncoder,
        validator: GroundedUseValidator,
        grouping_maps: Optional[FrozenGroupingMaps] = None,
        reward_fn: Optional[Callable[[str, str, Dict[str, Any]], float]] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.config = config
        self.config.validate()
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.validator = validator
        self.reward_fn = reward_fn

        if device is None:
            try:
                device = next(policy_model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        self.agent = MemoryPolicyAgent(
            model=policy_model,
            tokenizer=tokenizer,
            ref_model=ref_model,
            device=self.device,
            temperature=config.optimization.temperature,
            top_p=config.optimization.top_p,
            max_new_tokens_write=config.optimization.max_new_tokens_write,
            max_new_tokens_answer=config.optimization.max_new_tokens_answer,
        )

        self.grouping_maps = grouping_maps or FrozenGroupingMaps(config.grouping)
        self.router = ResponsibilityRouter(config.reward)
        self.group_builder = ItemUseGroupBuilder(
            config=config.grouping,
            grouping_maps=self.grouping_maps,
            seed=config.optimization.seed,
        )
        self.replay_queue = DelayedUseReplayQueue(
            config=config.replay,
            seed=config.optimization.seed,
        )

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        self.global_step: int = 0
        self.session_clock: int = 0

    def rollout_episode(
        self,
        episode: Episode,
        rollout_idx: int,
        do_sample: bool = True,
    ) -> RolloutTrace:
        """Execute one full rollout trajectory tau over a streaming/multi-session episode."""
        t_start = time.perf_counter()
        traj_id = f"{episode.episode_id}_r{rollout_idx}_s{self.global_step}"
        store = MemoryStore(
            max_memory_tokens=self.config.memory.max_memory_tokens,
            eviction_policy=self.config.memory.eviction_policy,
        )
        retriever = DenseMemoryRetriever(
            encoder=self.encoder,
            top_k=self.config.memory.top_k,
        )
        recorder = WRUTraceRecorder(
            trajectory_id=traj_id,
            input_group_id=episode.episode_id,
            benchmark=episode.benchmark,
        )

        total_gen_tokens = 0
        step_idx = 0

        # Phase 1: Stream incoming observations/session turns into persistent memory M_t.
        for chunk in episode.stream_units:
            step_idx += 1
            retrieved = retriever.retrieve(store, chunk.text, step=step_idx)
            action, write_span = self.agent.decide_memory_action(
                observation=chunk.text,
                retrieved=retrieved,
                do_sample=do_sample,
            )
            total_gen_tokens += write_span.response_length

            # Pre-write source embedding determines frozen item bucket phi(i) before write execution.
            pre_emb = self.encoder.encode_passages([chunk.text])[0]
            _, bucket_id = self.grouping_maps.assign_item_bucket(
                source_domain=chunk.source_domain,
                pre_write_embedding=pre_emb,
                fallback_text=chunk.text,
            )
            source_meta = SourceSpanMetadata(
                source_domain=chunk.source_domain,
                source_doc_id=chunk.source_doc_id,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                pre_write_segment=chunk.text,
                source_unit_id=chunk.unit_id,
            )
            transition = store.apply(
                operation=action.operation,
                content=action.content,
                step=step_idx,
                session=chunk.session_id,
                target_id=action.target_id,
                source_meta=source_meta,
                item_bucket=bucket_id,
            )
            write_ev = recorder.record_write(
                transition=transition,
                step=step_idx,
                session=chunk.session_id,
                source_domain=chunk.source_domain,
                item_bucket=bucket_id,
                token_span=write_span,
                metadata={
                    "source_doc_id": chunk.source_doc_id,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "source_unit_id": chunk.unit_id,
                },
            )
            if write_ev is not None and transition.item_id is not None:
                live_item = store.get(transition.item_id)
                if live_item is not None and live_item.active:
                    retriever.index_item(live_item)

        # Phase 2: Answer downstream requests using retrieved working sets S_q.
        for req in episode.requests:
            step_idx += 1
            retrieved = retriever.retrieve(store, req.question, step=step_idx)
            task_stratum = self.grouping_maps.assign_task_stratum(
                benchmark=episode.benchmark,
                request_type=req.request_type,
                context_length=req.context_length,
                ref_entropy=req.reference_entropy,
            )
            store_hash = store.exact_state_hash()

            recorder.record_retrieval(
                request_id=req.request_id,
                retrieved_items=[(r.item, r.rank, r.score) for r in retrieved],
                step=step_idx,
                session=req.session_id,
                task_stratum=task_stratum,
                store_hash=store_hash,
            )

            answer_text, answer_span = self.agent.generate_answer(
                question=req.question,
                retrieved=retrieved,
                do_sample=do_sample,
            )
            total_gen_tokens += answer_span.response_length

            # Extract citations and construct localized identifier-bearing token spans T(e).
            response_ids = answer_span.input_ids[answer_span.prompt_length :]
            citations = extract_citations_and_spans(answer_text)
            for cit in citations:
                cit_mask = build_token_span_mask(
                    tokenizer=self.tokenizer,
                    response_ids=response_ids,
                    char_start=cit.char_start,
                    char_end=cit.char_end,
                )
                cit_span = TokenSpanData(
                    input_ids=list(answer_span.input_ids),
                    prompt_length=answer_span.prompt_length,
                    responsible_mask=cit_mask,
                    behavior_logprobs=list(answer_span.behavior_logprobs),
                    ref_logprobs=list(answer_span.ref_logprobs) if answer_span.ref_logprobs is not None else None,
                )
                recorder.record_citation(
                    request_id=req.request_id,
                    item_id=cit.item_id,
                    cited_span=cit.supporting_span,
                    step=step_idx,
                    session=req.session_id,
                    answer_text=answer_text,
                    task_stratum=task_stratum,
                    store_hash=store_hash,
                    token_span=cit_span,
                )

            task_score = 0.0
            if self.reward_fn is not None:
                task_score = float(self.reward_fn(answer_text, req.answer, req.metadata))

            recorder.record_outcome(
                request_id=req.request_id,
                task_reward=task_score,
                answer_text=answer_text,
                step=step_idx,
                session=req.session_id,
                is_successful=task_score > 0.0,
                task_stratum=task_stratum,
                store_hash=store_hash,
                token_span=answer_span,
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return recorder.build_trace(
            store_tokens_at_end=store.active_token_count,
            total_generated_tokens=total_gen_tokens,
            serving_latency_ms=elapsed_ms,
        )

    def _evaluate_event_logprobs(self, event: WRUEvent) -> Sequence[float]:
        """Evaluate current policy log-probabilities for importance-ratio replay filtering."""
        span = event.token_span
        if not span.input_ids or span.response_length <= 0:
            return []
        input_ids = torch.tensor([span.input_ids], dtype=torch.long, device=self.device)
        self.policy_model.eval()
        with torch.no_grad():
            logprobs, _ = compute_sequence_logprobs(
                model=self.policy_model,
                input_ids=input_ids,
                prompt_length=span.prompt_length,
            )
        return [float(v) for v in logprobs.cpu().tolist()]

    def _assign_outcome_grpo(self, traces: Sequence[RolloutTrace]) -> List[ComparisonGroup]:
        """Baseline operator: Outcome GRPO trajectory broadcast (Equation 2)."""
        by_input: Dict[str, List[RolloutTrace]] = {}
        for tr in traces:
            by_input.setdefault(tr.input_group_id, []).append(tr)

        groups: List[ComparisonGroup] = []
        for idx, (group_key, group_traces) in enumerate(by_input.items()):
            rewards = [tr.terminal_reward for tr in group_traces]
            if len(rewards) < 2:
                advs = [0.0] * len(rewards)
            else:
                mean_r = sum(rewards) / len(rewards)
                var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
                std_r = var_r ** 0.5
                advs = [0.0] * len(rewards) if std_r <= self.config.reward.eps else [(r - mean_r) / std_r for r in rewards]

            all_evs: List[WRUEvent] = []
            for tr, traj_adv in zip(group_traces, advs):
                for ev in tr.events:
                    ev.adv_task = traj_adv
                    ev.adv_use = 0.0
                    ev.adv_cost = 0.0
                    ev.advantage = traj_adv
                    all_evs.append(ev)

            groups.append(
                ComparisonGroup(
                    group_id=f"grpo_{idx}",
                    group_kind="answer",
                    key=("OUTCOME_GRPO", group_key),
                    events=all_evs,
                )
            )
        return groups

    def _assign_write_local_contrast(
        self,
        current_events: Sequence[WRUEvent],
    ) -> tuple[List[ComparisonGroup], ChannelNormalizationStats]:
        """Table 14 operator: write-local contrast grouped by input episode."""
        by_input_stage: Dict[tuple[str, str], List[WRUEvent]] = {}
        for ev in current_events:
            by_input_stage.setdefault((ev.input_group_id, ev.event_type.value), []).append(ev)

        groups = [
            ComparisonGroup(
                group_id=f"write_local_{idx}",
                group_kind="answer" if k[1] == EventType.OUTCOME.value else "replay",
                key=k,
                events=evs,
            )
            for idx, (k, evs) in enumerate(by_input_stage.items())
        ]
        stats = normalize_and_combine_advantages(
            groups=groups,
            config=self.config.reward,
            min_group_size=self.config.grouping.min_group_size,
        )
        return groups, stats

    def train_on_traces(
        self,
        traces: Sequence[RolloutTrace],
        agentic_mode: bool = False,
        apply_optimizer_step: bool = True,
    ) -> TrainingStepMetrics:
        """Run lines 4-14 of Algorithm 1 on a collected batch of RolloutTraces.

        Separating trace processing from rollout execution enables both online
        training (`train_step`) and deterministic offline trace optimization.
        """
        t_trace_start = time.perf_counter()
        self.session_clock += 1

        # Lines 4-5: Validate citations (Eq. 6 / 19) and route multi-channel rewards (Eq. 7, 10, 11).
        current_events: List[WRUEvent] = []
        for trace in traces:
            self.validator.validate_trace(trace, agentic_mode=agentic_mode)
            self.router.route_trace(trace)
            current_events.extend(trace.events)

        # Lines 6-9: Archive writes and enqueue delayed-use write-use pairs into Q_delay.
        if self.config.replay.enabled:
            for ev in current_events:
                if ev.event_type == EventType.WRITE:
                    self.replay_queue.register_write(ev)
                    if not ev.is_retrieved:
                        # Unretrieved writes enter cost-only replay groups under NO-USE sentinels.
                        key = self.group_builder.compute_item_use_key(ev)
                        self.replay_queue.enqueue_delayed_pair(
                            origin_write=ev,
                            use_event=None,
                            grouping_key=key,
                            current_session=self.session_clock,
                        )
            for ev in current_events:
                if ev.event_type == EventType.CITE and ev.is_grounded and ev.item_id is not None:
                    origin_write = self.replay_queue.match_origin_write(ev.item_id)
                    if origin_write is not None:
                        write_key = self.group_builder.compute_item_use_key(origin_write)
                        self.replay_queue.enqueue_delayed_pair(
                            origin_write=origin_write,
                            use_event=ev,
                            grouping_key=write_key,
                            current_session=self.session_clock,
                        )

        operator = self.config.optimization.assignment_operator
        replay_events: List[WRUEvent] = []
        replay_diag = {"expired": 0, "ratio_discarded": 0, "sampled": 0}

        if operator == "outcome_grpo":
            groups = self._assign_outcome_grpo(traces)
            norm_stats = ChannelNormalizationStats(
                task_channel_sd=1.0,
                use_channel_sd=0.0,
                cost_channel_sd=0.0,
                usable_events=len(current_events),
                zero_variance_groups=0,
                singleton_groups=0,
            )
            all_opt_events = list(current_events)
        elif operator == "write_local":
            groups, norm_stats = self._assign_write_local_contrast(current_events)
            all_opt_events = list(current_events)
        else:
            # Sample matching replay entries with 32-session expiry and [0.5, 2.0] ratio cutoff.
            if self.config.replay.enabled and self.config.replay.use_replay_groups and operator != "exact_state":
                replay_events, replay_diag = self.replay_queue.sample_matching_replay(
                    current_events=current_events,
                    key_fn=self.group_builder.compute_item_use_key,
                    current_session=self.session_clock,
                    logprob_evaluator=self._evaluate_event_logprobs,
                )

            # Line 10: Construct disjoint listwise, replay, and answer comparison groups.
            prev_strategy = self.config.grouping.strategy
            if operator == "exact_state":
                self.config.grouping.strategy = "exact_state"
            groups = self.group_builder.build_groups(
                current_events=current_events,
                replay_events=replay_events,
                use_listwise=self.config.replay.use_listwise_groups,
                use_replay=self.config.replay.use_replay_groups,
            )
            self.config.grouping.strategy = prev_strategy

            # Lines 11-12: Channel-wise normalization and advantage aggregation (Eq. 9 & 12).
            norm_stats = normalize_and_combine_advantages(
                groups=groups,
                config=self.config.reward,
                min_group_size=self.config.grouping.min_group_size,
            )
            all_opt_events = [ev for grp in groups for ev in grp.events]

        group_diag = self.group_builder.group_diagnostics(
            groups, min_group_size=self.config.grouping.min_group_size
        )
        exact_cov = compute_exact_state_coverage(traces)
        trace_ms = (time.perf_counter() - t_trace_start) * 1000.0

        # Lines 13-14: Compute L_UGPO and update policy parameters theta.
        self.policy_model.train()
        loss_out: UGPOLossOutput = compute_ugpo_loss(
            policy_model=self.policy_model,
            ref_model=self.ref_model,
            events=all_opt_events,
            clip_delta=self.config.optimization.clip_delta,
            kl_beta=self.config.optimization.kl_beta,
            device=self.device,
        )

        if apply_optimizer_step and loss_out.num_responsible_tokens > 0:
            self.optimizer.zero_grad(set_to_none=True)
            loss_out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                max_norm=self.config.optimization.max_grad_norm,
            )
            self.optimizer.step()

        self.global_step += 1
        mean_reward = (
            float(sum(tr.terminal_reward for tr in traces) / len(traces)) if traces else 0.0
        )

        return TrainingStepMetrics(
            step=self.global_step,
            loss=float(loss_out.loss.detach().item()),
            policy_loss=loss_out.policy_loss,
            kl_loss=loss_out.kl_loss,
            clip_fraction=loss_out.clip_fraction,
            mean_ratio=loss_out.mean_ratio,
            mean_task_reward=mean_reward,
            non_singleton_coverage=group_diag["non_singleton_coverage"],
            exact_state_coverage=exact_cov,
            effective_group_size=group_diag["effective_group_size"],
            task_channel_sd=norm_stats.task_channel_sd,
            use_channel_sd=norm_stats.use_channel_sd,
            cost_channel_sd=norm_stats.cost_channel_sd,
            num_current_events=len(current_events),
            num_replay_events=len(replay_events),
            replay_expired=replay_diag["expired"],
            replay_ratio_discarded=replay_diag["ratio_discarded"],
            trace_processing_ms=trace_ms,
        )

    def train_step(
        self,
        episodes: Sequence[Episode],
        agentic_mode: bool = False,
    ) -> TrainingStepMetrics:
        """Execute one full end-to-end Algorithm 1 step over a batch of episodes."""
        traces: List[RolloutTrace] = []
        for episode in episodes:
            for r_idx in range(self.config.optimization.rollouts_per_input):
                trace = self.rollout_episode(episode, rollout_idx=r_idx, do_sample=True)
                traces.append(trace)
        return self.train_on_traces(traces, agentic_mode=agentic_mode, apply_optimizer_step=True)
