"""Clipped UGPO policy objective with responsible-token masking and KL regularization.

Implements Equations 13 and 14:
    rho_j(theta) = pi_theta(a_j | h_j) / pi_{b(e)}(a_j | h_j)
    L_UGPO(theta) = -E_{e, j in T(e)}[min(rho_j(theta) * A_hat_e,
                                          clip_{delta_clip}(rho_j(theta)) * A_hat_e)]
                    + beta * D_KL(pi_theta || pi_ref)

Applies item-use advantages strictly to the responsible token span T(e) of
each WRU lifecycle event and uses the non-negative unbiased k3 KL estimator
standard in group-relative policy optimization (Shao et al., 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ugpo.trace.schema import WRUEvent


@dataclass
class UGPOLossOutput:
    """Scalar loss tensor and optimization diagnostics from one UGPO objective pass."""

    loss: torch.Tensor
    policy_loss: float
    kl_loss: float
    clip_fraction: float
    mean_ratio: float
    num_events_used: int
    num_responsible_tokens: int


def _extract_logits(model_output: object) -> torch.Tensor:
    """Extract [batch, seq_len, vocab] logits from HuggingFace or raw PyTorch models."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (tuple, list)) and len(model_output) > 0:
        return model_output[0]
    raise TypeError(f"Unsupported model output type for logits extraction: {type(model_output)}")


def compute_sequence_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_length: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log pi(a_j | h_j) and token entropy over generated tokens.

    Args:
        model: Causal language model accepting `input_ids` (and optional `attention_mask`).
        input_ids: LongTensor of shape [1, seq_len] containing prompt + generated tokens.
        prompt_length: Number of prompt tokens preceding the generated response.
        attention_mask: Optional LongTensor of shape [1, seq_len].

    Returns:
        Tuple `(token_logprobs, token_entropies)` each of shape [response_len],
        aligned with `input_ids[0, prompt_length:]`.
    """
    seq_len = int(input_ids.shape[1])
    response_len = seq_len - prompt_length
    if response_len <= 0:
        empty = torch.zeros(0, device=input_ids.device, dtype=torch.float32)
        return empty, empty

    if attention_mask is not None:
        try:
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            output = model(input_ids)
    else:
        try:
            output = model(input_ids=input_ids)
        except TypeError:
            output = model(input_ids)

    logits = _extract_logits(output)  # [1, seq_len, vocab]
    # Position t predicts token t+1; generated tokens start at prompt_length.
    start_pos = max(0, prompt_length - 1)
    shift_logits = logits[0, start_pos : seq_len - 1, :].float()
    target_start = start_pos + 1
    target_ids = input_ids[0, target_start:seq_len]

    log_probs_vocab = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs_vocab.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    probs_vocab = log_probs_vocab.exp()
    token_entropies = -(probs_vocab * log_probs_vocab).sum(dim=-1)

    # If prompt_length == 0, pad leading token with zero so alignment matches input_ids[0, 0:].
    if prompt_length == 0:
        pad = torch.zeros(1, device=input_ids.device, dtype=torch.float32)
        token_logprobs = torch.cat([pad, token_logprobs], dim=0)
        token_entropies = torch.cat([pad, token_entropies], dim=0)

    return token_logprobs, token_entropies


def compute_ugpo_loss(
    policy_model: nn.Module,
    ref_model: Optional[nn.Module],
    events: Sequence[WRUEvent],
    clip_delta: float = 0.2,
    kl_beta: float = 0.01,
    device: Optional[torch.device | str] = None,
) -> UGPOLossOutput:
    """Compute Equations 13-14 over a collection of advantage-annotated WRU events.

    Only events with at least one responsible token in T(e) and a non-empty
    token sequence participate in the objective. Stored behavior log-probs
    `token_span.behavior_logprobs` define pi_{b(e)}, enabling seamless
    off-policy importance weighting for delayed-use replay events.
    """
    if device is None:
        try:
            device = next(policy_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    total_policy_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_kl_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens = 0
    total_clipped = 0
    ratio_sum = 0.0
    events_used = 0

    for ev in events:
        span = ev.token_span
        if not span.input_ids or span.response_length <= 0:
            continue
        if not span.responsible_mask or not any(span.responsible_mask):
            continue

        input_ids = torch.tensor([span.input_ids], dtype=torch.long, device=device)
        cur_logprobs, _ = compute_sequence_logprobs(
            model=policy_model,
            input_ids=input_ids,
            prompt_length=span.prompt_length,
        )

        resp_len = int(cur_logprobs.shape[0])
        if resp_len == 0:
            continue

        # Align responsible mask T(e) to response length.
        mask_list = span.responsible_mask[:resp_len]
        if len(mask_list) < resp_len:
            mask_list = mask_list + [False] * (resp_len - len(mask_list))
        mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
        n_resp = int(mask.sum().item())
        if n_resp == 0:
            continue

        cur_lp = cur_logprobs[mask]

        # Behavior policy log-probabilities log pi_{b(e)}(a_j | h_j).
        if span.behavior_logprobs and len(span.behavior_logprobs) >= resp_len:
            beh_all = torch.tensor(span.behavior_logprobs[:resp_len], dtype=torch.float32, device=device)
            beh_lp = beh_all[mask]
        else:
            beh_lp = cur_lp.detach()

        # Equation 13: importance ratio rho_j(theta) = pi_theta / pi_{b(e)}.
        log_ratio = (cur_lp - beh_lp).clamp(min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_delta, 1.0 + clip_delta)

        adv = torch.tensor(float(ev.advantage), dtype=torch.float32, device=device)
        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        event_pg_loss = -torch.minimum(surr1, surr2).sum()

        # KL regularizer D_KL(pi_theta || pi_ref) using non-negative k3 estimator.
        if kl_beta > 0.0 and (ref_model is not None or span.ref_logprobs is not None):
            if span.ref_logprobs is not None and len(span.ref_logprobs) >= resp_len:
                ref_lp = torch.tensor(span.ref_logprobs[:resp_len], dtype=torch.float32, device=device)[mask]
            else:
                with torch.no_grad():
                    ref_all, _ = compute_sequence_logprobs(
                        model=ref_model,
                        input_ids=input_ids,
                        prompt_length=span.prompt_length,
                    )
                    ref_lp = ref_all[mask]
            ref_log_diff = (ref_lp - cur_lp).clamp(min=-20.0, max=20.0)
            kl_per_token = torch.exp(ref_log_diff) - ref_log_diff - 1.0
            event_kl = kl_per_token.sum()
        else:
            event_kl = torch.zeros((), device=device, dtype=torch.float32)

        total_policy_loss = total_policy_loss + event_pg_loss
        total_kl_loss = total_kl_loss + event_kl
        total_tokens += n_resp
        total_clipped += int(((ratio < 1.0 - clip_delta) | (ratio > 1.0 + clip_delta)).sum().item())
        ratio_sum += float(ratio.detach().sum().item())
        events_used += 1

    if total_tokens == 0:
        zero = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
        return UGPOLossOutput(
            loss=zero,
            policy_loss=0.0,
            kl_loss=0.0,
            clip_fraction=0.0,
            mean_ratio=1.0,
            num_events_used=0,
            num_responsible_tokens=0,
        )

    mean_pg_loss = total_policy_loss / float(total_tokens)
    mean_kl_loss = total_kl_loss / float(total_tokens)
    total_loss = mean_pg_loss + kl_beta * mean_kl_loss

    return UGPOLossOutput(
        loss=total_loss,
        policy_loss=float(mean_pg_loss.detach().item()),
        kl_loss=float(mean_kl_loss.detach().item()),
        clip_fraction=float(total_clipped / total_tokens),
        mean_ratio=float(ratio_sum / total_tokens),
        num_events_used=events_used,
        num_responsible_tokens=total_tokens,
    )
