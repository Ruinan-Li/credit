"""Memory-control policy agent wrapper for Qwen2.5-Instruct backbones.

Handles:
- Structured memory-control prompting and action parsing (ADD, UPDATE, DELETE, NOOP).
- Grounded response generation with stable identifier citations `[mem_XXXXXX]`.
- ReAct think-act prompting and parsing for agentic transfer (ALFWorld / WebShop).
- Responsible token-span mask extraction T(e) (Appendix E).
- Frozen reference-policy greedy entropy computation for difficulty quintiles psi(q).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ugpo.memory.retriever import RetrievedItem
from ugpo.memory.store import MemoryOperation
from ugpo.policy.loss import compute_sequence_logprobs
from ugpo.trace.schema import TokenSpanData


_OP_PATTERN = re.compile(
    r"OPERATION\s*:\s*(ADD|UPDATE|DELETE|NOOP)",
    re.IGNORECASE,
)
_TARGET_PATTERN = re.compile(
    r"TARGET_ID\s*:\s*([A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)
_CONTENT_PATTERN = re.compile(
    r"CONTENT\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
_THOUGHT_PATTERN = re.compile(
    r"Thought\s*:\s*(.+?)(?=\nAction\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_ACTION_PATTERN = re.compile(
    r"Action\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class MemoryAction:
    """Parsed structured memory-control decision (o_t, v_t) from Equation 1."""

    operation: MemoryOperation
    content: str = ""
    target_id: Optional[str] = None
    raw_text: str = ""


@dataclass
class ReActAction:
    """Parsed ReAct thought-action step for ALFWorld and WebShop (Appendix H)."""

    thought: str
    action: str
    raw_text: str = ""


def parse_memory_action(raw_text: str) -> MemoryAction:
    """Parse model output into a structured ADD / UPDATE / DELETE / NOOP action."""
    text = raw_text.strip()
    op_match = _OP_PATTERN.search(text)
    if not op_match:
        return MemoryAction(operation=MemoryOperation.NOOP, content="", target_id=None, raw_text=text)

    op = MemoryOperation(op_match.group(1).upper())
    target_match = _TARGET_PATTERN.search(text)
    target_id = target_match.group(1).strip() if target_match else None
    if target_id and target_id.upper() in ("NONE", "NULL", "N/A"):
        target_id = None

    content_match = _CONTENT_PATTERN.search(text)
    content = content_match.group(1).strip() if content_match else ""
    return MemoryAction(operation=op, content=content, target_id=target_id, raw_text=text)


def parse_react_action(raw_text: str) -> ReActAction:
    """Parse a ReAct Thought / Action block for agentic transfer tasks."""
    text = raw_text.strip()
    t_match = _THOUGHT_PATTERN.search(text)
    a_match = _ACTION_PATTERN.search(text)
    thought = t_match.group(1).strip() if t_match else ""
    action = a_match.group(1).strip().splitlines()[0].strip() if a_match else text.splitlines()[-1].strip()
    return ReActAction(thought=thought, action=action, raw_text=text)


def build_token_span_mask(
    tokenizer: Any,
    response_ids: Sequence[int],
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
) -> List[bool]:
    """Build a boolean responsible-token mask T(e) over generated response tokens.

    If `char_start` and `char_end` are omitted, marks all generated tokens as
    responsible (matching write-span and full-answer events in Appendix E).
    When a character span is specified (e.g., an identifier-bearing citation
    span), selects the exact response tokens intersecting `[char_start, char_end)`.
    """
    n_tokens = len(response_ids)
    if n_tokens == 0:
        return []
    if char_start is None or char_end is None:
        return [True] * n_tokens

    mask = [False] * n_tokens
    cursor = 0
    for idx, tok_id in enumerate(response_ids):
        piece = tokenizer.decode([int(tok_id)], skip_special_tokens=False)
        piece_start = cursor
        piece_end = cursor + len(piece)
        cursor = piece_end
        if piece_end > char_start and piece_start < char_end:
            mask[idx] = True

    if not any(mask):
        return [True] * n_tokens
    return mask


class MemoryPolicyAgent:
    """Policy wrapper supporting memory updates, grounded QA, and ReAct actions.

    Designed for Qwen2.5-7B/14B/32B-Instruct backbones with 128K read context
    and works with any HuggingFace causal language model and tokenizer.
    """

    WRITE_SYSTEM_PROMPT = (
        "You manage an external long-term memory store. Given the incoming observation "
        "and currently retrieved memory items, decide whether to ADD a new concise fact, "
        "UPDATE an existing item, DELETE an obsolete item, or emit NOOP.\n"
        "Respond strictly in the format:\n"
        "OPERATION: <ADD|UPDATE|DELETE|NOOP>\n"
        "TARGET_ID: <item_id or NONE>\n"
        "CONTENT: <memory text or empty>"
    )

    ANSWER_SYSTEM_PROMPT = (
        "Answer the user request using the retrieved memory items below. "
        "Whenever a sentence relies on a memory item, cite its exact identifier in "
        "square brackets, e.g. [mem_000001]. Only cite items that directly support the sentence."
    )

    REACT_SYSTEM_PROMPT = (
        "You are an interactive agent with external memory. Think step-by-step, "
        "cite any retrieved memory item identifier [mem_XXXXXX] used in your reasoning, "
        "and emit the next environment command.\n"
        "Format:\n"
        "Thought: <reasoning citing [mem_XXXXXX] when applicable>\n"
        "Action: <single executable command>"
    )

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        ref_model: Optional[nn.Module] = None,
        device: Optional[torch.device | str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens_write: int = 256,
        max_new_tokens_answer: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens_write = max_new_tokens_write
        self.max_new_tokens_answer = max_new_tokens_answer

    @staticmethod
    def format_retrieved_context(retrieved: Sequence[RetrievedItem]) -> str:
        """Render top-k retrieved items with their stable identifiers."""
        if not retrieved:
            return "(empty memory context)"
        lines = []
        for entry in retrieved:
            lines.append(f"[{entry.item.item_id}] {entry.item.content}")
        return "\n".join(lines)

    def build_write_prompt(self, observation: str, retrieved: Sequence[RetrievedItem]) -> str:
        """Construct the memory-control prompt for state s_t = (x_t, S_t, M_{t-1})."""
        mem_block = self.format_retrieved_context(retrieved)
        return (
            f"{self.WRITE_SYSTEM_PROMPT}\n\n"
            f"Retrieved Memory:\n{mem_block}\n\n"
            f"Observation:\n{observation}\n\n"
            f"Decision:\n"
        )

    def build_answer_prompt(self, question: str, retrieved: Sequence[RetrievedItem]) -> str:
        """Construct the grounded answer prompt for downstream request q."""
        mem_block = self.format_retrieved_context(retrieved)
        return (
            f"{self.ANSWER_SYSTEM_PROMPT}\n\n"
            f"Retrieved Memory:\n{mem_block}\n\n"
            f"Request:\n{question}\n\n"
            f"Answer:\n"
        )

    def build_react_prompt(
        self,
        goal: str,
        observation: str,
        retrieved: Sequence[RetrievedItem],
    ) -> str:
        """Construct a ReAct prompt for ALFWorld or WebShop."""
        mem_block = self.format_retrieved_context(retrieved)
        return (
            f"{self.REACT_SYSTEM_PROMPT}\n\n"
            f"Goal: {goal}\n"
            f"Retrieved Memory:\n{mem_block}\n\n"
            f"Observation: {observation}\n"
        )

    def generate_with_logprobs(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool = True,
    ) -> Tuple[str, TokenSpanData]:
        """Sample response tokens and record exact behavior-policy log-probabilities."""
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        prompt_ids = encoded["input_ids"].to(self.device)
        prompt_len = int(prompt_ids.shape[1])

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None)
            or getattr(self.tokenizer, "eos_token_id", 0),
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        self.model.eval()
        with torch.no_grad():
            full_ids = self.model.generate(prompt_ids, **gen_kwargs)
            beh_logprobs, _ = compute_sequence_logprobs(
                model=self.model,
                input_ids=full_ids,
                prompt_length=prompt_len,
            )
            ref_logprobs_list: Optional[List[float]] = None
            if self.ref_model is not None:
                ref_lp, _ = compute_sequence_logprobs(
                    model=self.ref_model,
                    input_ids=full_ids,
                    prompt_length=prompt_len,
                )
                ref_logprobs_list = [float(v) for v in ref_lp.cpu().tolist()]

        response_ids = full_ids[0, prompt_len:].cpu().tolist()
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        span = TokenSpanData(
            input_ids=[int(x) for x in full_ids[0].cpu().tolist()],
            prompt_length=prompt_len,
            responsible_mask=[True] * len(response_ids),
            behavior_logprobs=[float(v) for v in beh_logprobs.cpu().tolist()],
            ref_logprobs=ref_logprobs_list,
        )
        return response_text, span

    def decide_memory_action(
        self,
        observation: str,
        retrieved: Sequence[RetrievedItem],
        do_sample: bool = True,
    ) -> Tuple[MemoryAction, TokenSpanData]:
        """Sample a memory-control operation and return parsed action + token span."""
        prompt = self.build_write_prompt(observation, retrieved)
        raw_text, span = self.generate_with_logprobs(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens_write,
            do_sample=do_sample,
        )
        return parse_memory_action(raw_text), span

    def generate_answer(
        self,
        question: str,
        retrieved: Sequence[RetrievedItem],
        do_sample: bool = True,
    ) -> Tuple[str, TokenSpanData]:
        """Generate a grounded response with `[item_id]` citations and token span."""
        prompt = self.build_answer_prompt(question, retrieved)
        return self.generate_with_logprobs(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens_answer,
            do_sample=do_sample,
        )

    def decide_react_action(
        self,
        goal: str,
        observation: str,
        retrieved: Sequence[RetrievedItem],
        do_sample: bool = True,
    ) -> Tuple[ReActAction, TokenSpanData]:
        """Generate a ReAct thought-action step for ALFWorld / WebShop."""
        prompt = self.build_react_prompt(goal, observation, retrieved)
        raw_text, span = self.generate_with_logprobs(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens_answer,
            do_sample=do_sample,
        )
        return parse_react_action(raw_text), span

    def compute_reference_entropy(
        self,
        prompt: str,
        max_new_tokens: int = 64,
    ) -> float:
        """Compute mean next-token entropy along a greedy response of frozen pi_ref.

        Used once prior to optimization to construct the reference-policy
        difficulty quintile in psi(q) without task labels (Appendix E).
        """
        ref = self.ref_model if self.ref_model is not None else self.model
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        prompt_ids = encoded["input_ids"].to(self.device)
        prompt_len = int(prompt_ids.shape[1])

        ref.eval()
        with torch.no_grad():
            full_ids = ref.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None)
                or getattr(self.tokenizer, "eos_token_id", 0),
            )
            _, entropies = compute_sequence_logprobs(
                model=ref,
                input_ids=full_ids,
                prompt_length=prompt_len,
            )
        if int(entropies.numel()) == 0:
            return 0.0
        return float(entropies.mean().item())
