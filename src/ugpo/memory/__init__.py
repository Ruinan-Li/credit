"""Persistent external memory store and dense retrieval modules."""

from ugpo.memory.retriever import DenseMemoryRetriever, E5EmbeddingEncoder, LexicalEmbeddingEncoder
from ugpo.memory.store import MemoryItem, MemoryOperation, MemoryStore, MemoryTransitionResult

__all__ = [
    "DenseMemoryRetriever",
    "E5EmbeddingEncoder",
    "LexicalEmbeddingEncoder",
    "MemoryItem",
    "MemoryOperation",
    "MemoryStore",
    "MemoryTransitionResult",
]
