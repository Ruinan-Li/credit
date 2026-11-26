"""PyTorch gradient, clipping, KL, and Algorithm 1 training step tests."""

from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn

from ugpo.config import GroundingConfig, UGPOConfig
from ugpo.memory.retriever import LexicalEmbeddingEncoder
from ugpo.policy.agent import parse_memory_action, parse_react_action
from ugpo.policy.loss import compute_sequence_logprobs, compute_ugpo_loss
from ugpo.policy.trainer import UGPOTrainer
from ugpo.trace.grounding import GroundedUseValidator, TokenOverlapVerifier
from ugpo.trace.schema import EventType, RolloutTrace, TokenSpanData, WRUEvent


class CompactCausalLM(nn.Module):
    """Lightweight causal Transformer language model for PyTorch optimization tests."""

    def __init__(self, vocab_size: int = 128, d_model: int = 32, nhead: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(256, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        hidden = self.embed(input_ids) + self.pos(positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        out = self.encoder(hidden, mask=causal_mask)
        return self.lm_head(out)


class WhitespaceTokenizer:
    """Deterministic vocabulary tokenizer for self-contained PyTorch tests."""

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True) -> Dict[str, torch.Tensor]:
        ids = [2 + (sum(ord(ch) for ch in tok) % (self.vocab_size - 2)) for tok in text.split()[:32]]
        if not ids:
            ids = [2]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{tid}" for tid in token_ids)


def test_action_parsers() -> None:
    action = parse_memory_action("OPERATION: UPDATE\nTARGET_ID: mem_000003\nCONTENT: Updated birth year 1966")
    assert action.operation.value == "UPDATE"
    assert action.target_id == "mem_000003"
    assert action.content == "Updated birth year 1966"

    react = parse_react_action("Thought: The mug is on countertop [mem_000002].\nAction: go to countertop 1")
    assert "[mem_000002]" in react.thought
    assert react.action == "go to countertop 1"


def test_ugpo_loss_gradients_and_token_masking() -> None:
    torch.manual_seed(0)
    policy = CompactCausalLM(vocab_size=64, d_model=16, nhead=2)
    ref = copy.deepcopy(policy)
    ref.eval()

    seq = torch.tensor([[3, 5, 7, 11, 13, 17, 19]], dtype=torch.long)
    prompt_len = 3
    with torch.no_grad():
        beh_lp, _ = compute_sequence_logprobs(policy, seq, prompt_length=prompt_len)

    ev_pos = WRUEvent(
        event_id="e1",
        event_type=EventType.WRITE,
        item_id="m1",
        step=1,
        advantage=1.2,
        token_span=TokenSpanData(
            input_ids=seq[0].tolist(),
            prompt_length=prompt_len,
            responsible_mask=[True, True, False, False],
            behavior_logprobs=beh_lp.tolist(),
        ),
    )
    ev_neg = WRUEvent(
        event_id="e2",
        event_type=EventType.WRITE,
        item_id="m2",
        step=2,
        advantage=-0.8,
        token_span=TokenSpanData(
            input_ids=seq[0].tolist(),
            prompt_length=prompt_len,
            responsible_mask=[False, False, True, True],
            behavior_logprobs=beh_lp.tolist(),
        ),
    )

    out = compute_ugpo_loss(policy, ref, [ev_pos, ev_neg], clip_delta=0.2, kl_beta=0.01)
    assert out.num_events_used == 2
    assert out.num_responsible_tokens == 4
    assert abs(out.mean_ratio - 1.0) < 1e-4
    assert abs(out.kl_loss) < 1e-5

    out.loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_trainer_algorithm_1_step_updates_parameters() -> None:
    torch.manual_seed(42)
    policy = CompactCausalLM(vocab_size=64, d_model=16, nhead=2)
    ref = copy.deepcopy(policy)
    tokenizer = WhitespaceTokenizer(vocab_size=64)

    cfg = UGPOConfig()
    cfg.grounding = GroundingConfig(verifier_mode="token_overlap", token_overlap_threshold=0.4)
    cfg.optimization.learning_rate = 1e-3

    validator = GroundedUseValidator(cfg.grounding, verifier=TokenOverlapVerifier())
    trainer = UGPOTrainer(
        config=cfg,
        policy_model=policy,
        ref_model=ref,
        tokenizer=tokenizer,
        encoder=LexicalEmbeddingEncoder(dim=64),
        validator=validator,
    )

    # Construct two rollouts for the same episode with distinct grounded outcomes.
    traces: List[RolloutTrace] = []
    for idx, (grounded, reward) in enumerate([(True, 1.0), (False, 0.0)]):
        seq = torch.tensor([[4, 6, 8, 10 + idx, 12 + idx, 14, 16]], dtype=torch.long)
        with torch.no_grad():
            beh_lp, _ = compute_sequence_logprobs(policy, seq, prompt_length=3)
        span = TokenSpanData(
            input_ids=seq[0].tolist(),
            prompt_length=3,
            responsible_mask=[True, True, True, True],
            behavior_logprobs=beh_lp.tolist(),
        )
        item_id = f"mem_{idx:06d}"
        w_ev = WRUEvent(
            event_id=f"w_{idx}",
            event_type=EventType.WRITE,
            item_id=item_id,
            step=1,
            session=0,
            trajectory_id=f"traj_{idx}",
            input_group_id="ep_0",
            operation="ADD",
            item_content="Scott Derrickson is an American director",
            item_tokens=6 + idx * 4,
            store_hash=f"hash_{idx}",
            item_bucket=0,
            write_step=1,
            write_session=0,
            token_span=copy.deepcopy(span),
        )
        r_ev = WRUEvent(
            event_id=f"r_{idx}",
            event_type=EventType.RETRIEVE,
            item_id=item_id,
            step=2,
            session=1,
            request_id="q0",
            rank=1,
            trajectory_id=f"traj_{idx}",
            input_group_id="ep_0",
            item_content=w_ev.item_content,
            item_tokens=w_ev.item_tokens,
            store_hash=f"hash_{idx}",
            item_bucket=0,
            task_stratum=("hotpotqa", "comparison", 0, 0),
            write_step=1,
            write_session=0,
            is_retrieved=True,
            token_span=copy.deepcopy(span),
        )
        c_ev = WRUEvent(
            event_id=f"c_{idx}",
            event_type=EventType.CITE,
            item_id=item_id,
            step=2,
            session=1,
            request_id="q0",
            trajectory_id=f"traj_{idx}",
            input_group_id="ep_0",
            item_content=w_ev.item_content,
            cited_span="Scott Derrickson is an American director" if grounded else "Unrelated claim about Mars 2099",
            store_hash=f"hash_{idx}",
            item_bucket=0,
            task_stratum=("hotpotqa", "comparison", 0, 0),
            write_step=1,
            write_session=0,
            is_retrieved=True,
            is_cited=True,
            token_span=copy.deepcopy(span),
        )
        o_ev = WRUEvent(
            event_id=f"o_{idx}",
            event_type=EventType.OUTCOME,
            item_id=None,
            step=2,
            session=1,
            request_id="q0",
            trajectory_id=f"traj_{idx}",
            input_group_id="ep_0",
            store_hash=f"hash_{idx}",
            task_stratum=("hotpotqa", "comparison", 0, 0),
            task_reward_raw=reward,
            is_successful_answer=reward > 0.0,
            token_span=copy.deepcopy(span),
        )
        traces.append(RolloutTrace(f"traj_{idx}", "ep_0", events=[w_ev, r_ev, c_ev, o_ev], terminal_reward=reward))

    before_weight = policy.lm_head.weight.detach().clone()
    metrics = trainer.train_on_traces(traces)
    after_weight = policy.lm_head.weight.detach()

    assert metrics.step == 1
    assert metrics.non_singleton_coverage > 0.0
    assert not torch.allclose(before_weight, after_weight)
