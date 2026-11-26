"""Dense memory retrieval backed by frozen E5-large embeddings.

Matches the retrieval interface described in Section 3.1 and Table 5:
given a downstream request query, returns the ranked top-k live item
identifiers S_q together with similarity scores and retrieval ranks.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from ugpo.memory.store import MemoryItem, MemoryStore


class TextEncoder(Protocol):
    """Minimal protocol implemented by dense embedding encoders."""

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode query strings into L2-normalized dense vectors."""

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Encode memory item passages into L2-normalized dense vectors."""


@dataclass
class RetrievedItem:
    """Single ranked candidate item returned for a downstream request."""

    item: MemoryItem
    rank: int
    score: float


class E5EmbeddingEncoder:
    """Frozen intfloat/e5-large-v2 encoder for retrieval and item bucketing.

    Follows the official E5 prefix convention ("query: " for queries and
    "passage: " for memory passages) with mean pooling and L2 normalization.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _encode_prefixed(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        prefixed = [f"{prefix}{t}" for t in texts]
        outputs: List[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(prefixed), self.batch_size):
                batch = prefixed[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                normed = self._torch.nn.functional.normalize(pooled, p=2, dim=-1)
                outputs.append(normed.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode_prefixed(texts, prefix="query: ")

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode_prefixed(texts, prefix="passage: ")


class LexicalEmbeddingEncoder:
    """Deterministic hashed-ngram dense encoder for CPU-only execution.

    Projects lowercased unigrams and bigrams into a fixed-dimensional
    L2-normalized feature space via signed feature hashing, requiring no
    external checkpoint download while preserving exact cosine retrieval
    and K-Means clustering geometry.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _encode_single(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec

        features = list(tokens)
        for idx in range(len(tokens) - 1):
            features.append(f"{tokens[idx]}_{tokens[idx + 1]}")

        counts: Dict[str, int] = {}
        for feat in features:
            counts[feat] = counts.get(feat, 0) + 1

        for feat, count in counts.items():
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            bucket = raw % self.dim
            sign = 1.0 if ((raw >> 63) & 1) == 0 else -1.0
            vec[bucket] += sign * (1.0 + math.log(count))

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._encode_single(t) for t in texts], axis=0)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_queries(texts)


class DenseMemoryRetriever:
    """Top-k dense retriever over live items in a MemoryStore.

    Caches passage embeddings keyed by stable item identifier so each
    materialized item is encoded at most once across retrieval requests.
    """

    def __init__(self, encoder: TextEncoder, top_k: int = 8) -> None:
        self.encoder = encoder
        self.top_k = top_k
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def index_item(self, item: MemoryItem) -> np.ndarray:
        """Encode and cache the embedding of a newly materialized memory item."""
        if item.item_id not in self._embedding_cache:
            emb = self.encoder.encode_passages([item.content])[0]
            self._embedding_cache[item.item_id] = emb
        return self._embedding_cache[item.item_id]

    def clear_cache(self) -> None:
        """Reset cached passage embeddings when starting a fresh episode."""
        self._embedding_cache.clear()

    def retrieve(
        self,
        store: MemoryStore,
        query: str,
        top_k: Optional[int] = None,
        step: Optional[int] = None,
    ) -> List[RetrievedItem]:
        """Return up to top_k active items ranked by cosine similarity to query."""
        k = self.top_k if top_k is None else top_k
        active = store.active_items
        if not active or k <= 0:
            return []

        missing = [item for item in active if item.item_id not in self._embedding_cache]
        if missing:
            embeddings = self.encoder.encode_passages([item.content for item in missing])
            for item, emb in zip(missing, embeddings):
                self._embedding_cache[item.item_id] = emb

        matrix = np.stack([self._embedding_cache[item.item_id] for item in active], axis=0)
        q_vec = self.encoder.encode_queries([query])[0]
        scores = matrix @ q_vec

        limit = min(k, len(active))
        top_indices = np.argsort(-scores, kind="mergesort")[:limit]

        results: List[RetrievedItem] = []
        retrieved_ids: List[str] = []
        for rank_idx, item_idx in enumerate(top_indices, start=1):
            item = active[int(item_idx)]
            retrieved_ids.append(item.item_id)
            results.append(
                RetrievedItem(
                    item=item,
                    rank=rank_idx,
                    score=float(scores[int(item_idx)]),
                )
            )

        if step is not None:
            store.mark_accessed(retrieved_ids, step=step)

        return results
