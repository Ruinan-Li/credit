"""Unit tests for persistent MemoryStore, DenseMemoryRetriever, WRUTraceRecorder, and GroundedUseValidator."""

from __future__ import annotations

from ugpo.config import GroundingConfig
from ugpo.memory.retriever import DenseMemoryRetriever, LexicalEmbeddingEncoder
from ugpo.memory.store import MemoryOperation, MemoryStore, SourceSpanMetadata
from ugpo.trace.grounding import GroundedUseValidator, TokenOverlapVerifier, extract_citations_and_spans
from ugpo.trace.recorder import WRUTraceRecorder
from ugpo.trace.schema import EventType


def test_memory_store_lineage_budget_and_exact_hash() -> None:
    store = MemoryStore(max_memory_tokens=15, eviction_policy="lru")

    r1 = store.apply(
        MemoryOperation.ADD,
        content="Caroline attended the LGBTQ support group on 7 May 2023",
        step=1,
        session=1,
        source_meta=SourceSpanMetadata(source_domain="dialogue", source_doc_id="D1:3"),
        item_bucket=2,
    )
    assert r1.item_id is not None
    hash_after_add = r1.store_hash

    r2 = store.apply(
        MemoryOperation.UPDATE,
        content="Caroline went to the LGBTQ support group on 7 May 2023 in Seattle",
        step=2,
        session=2,
        target_id=r1.item_id,
    )
    assert r2.item_id is not None
    assert r2.predecessor_id == r1.item_id
    assert r2.store_hash != hash_after_add
    assert store.resolve_latest_live_id(r1.item_id) == r2.item_id
    assert store.get_lineage_chain(r2.item_id) == [r1.item_id, r2.item_id]
    # Updated item inherits source metadata and frozen bucket from predecessor (Appendix E).
    updated = store.get(r2.item_id)
    assert updated is not None and updated.item_bucket == 2
    assert updated.source_meta.source_doc_id == "D1:3"

    # Budget eviction enforces 15-token cap when adding a large unrelated item.
    r3 = store.apply(
        MemoryOperation.ADD,
        content="Scott Derrickson is an American director screenwriter and producer born in Denver",
        step=3,
        session=3,
    )
    assert r3.item_id is not None
    assert store.active_token_count <= 15
    assert r2.item_id in r3.evicted_ids

    # Delete targets latest live item and removes it from active set.
    r4 = store.apply(MemoryOperation.DELETE, step=4, session=4, target_id=r3.item_id)
    assert r4.item_id == r3.item_id
    assert len(store.active_items) == 0


def test_dense_retrieval_and_wru_grounded_use() -> None:
    store = MemoryStore(max_memory_tokens=8192)
    encoder = LexicalEmbeddingEncoder(dim=256)
    retriever = DenseMemoryRetriever(encoder=encoder, top_k=2)
    recorder = WRUTraceRecorder(trajectory_id="traj_1", input_group_id="ep_1", benchmark="hotpotqa")

    t1 = store.apply(
        MemoryOperation.ADD,
        content="Scott Derrickson is an American film director and screenwriter.",
        step=1,
        session=0,
        item_bucket=1,
    )
    t2 = store.apply(
        MemoryOperation.ADD,
        content="Ed Wood was an American filmmaker, actor, and pulp novelist.",
        step=2,
        session=0,
        item_bucket=1,
    )
    t3 = store.apply(
        MemoryOperation.ADD,
        content="Mount Everest is located in the Mahalangur Himal sub-range of the Himalayas.",
        step=3,
        session=0,
        item_bucket=4,
    )
    for step, tr in enumerate([t1, t2, t3], start=1):
        recorder.record_write(tr, step=step, session=0, source_domain="wikipedia", item_bucket=1)
        assert tr.item_id is not None
        item = store.get(tr.item_id)
        assert item is not None
        retriever.index_item(item)

    retrieved = retriever.retrieve(store, query="Were Scott Derrickson and Ed Wood both American?", step=4)
    retrieved_ids = [r.item.item_id for r in retrieved]
    assert t1.item_id in retrieved_ids
    assert t2.item_id in retrieved_ids

    recorder.record_retrieval(
        request_id="q0",
        retrieved_items=[(r.item, r.rank, r.score) for r in retrieved],
        step=4,
        session=1,
        store_hash=store.exact_state_hash(),
    )

    answer = (
        f"Scott Derrickson is an American film director [{t1.item_id}]. "
        f"Ed Wood was born in Tokyo in 1999 [{t2.item_id}]. "
        f"[{t3.item_id}]"
    )
    citations = extract_citations_and_spans(answer)
    assert len(citations) == 3

    for cit in citations:
        recorder.record_citation(
            request_id="q0",
            item_id=cit.item_id,
            cited_span=cit.supporting_span,
            step=4,
            session=1,
            answer_text=answer,
        )
    recorder.record_outcome(request_id="q0", task_reward=1.0, answer_text=answer, step=4, session=1)

    trace = recorder.build_trace()
    validator = GroundedUseValidator(
        config=GroundingConfig(verifier_mode="token_overlap", token_overlap_threshold=0.5),
        verifier=TokenOverlapVerifier(),
    )
    decisions = validator.validate_trace(trace)

    # t1 is retrieved, cited with localized span, and entailed -> u_{q,i} == 1.
    assert decisions[("q0", t1.item_id)].u_qi == 1
    # t2 is retrieved and cited, but contradictory year/place fails entailment -> u_{q,i} == 0.
    assert decisions[("q0", t2.item_id)].u_qi == 0
    # t3 was neither retrieved nor given a localized claim span -> rejected before scoring.
    assert decisions[("q0", t3.item_id)].u_qi == 0
    assert not decisions[("q0", t3.item_id)].has_localized_span

    writes = {ev.item_id: ev for ev in trace.events_by_type(EventType.WRITE)}
    assert writes[t1.item_id].is_grounded is True
    assert writes[t2.item_id].is_grounded is False
    assert writes[t3.item_id].is_retrieved is False
