"""Benchmark loaders for HotpotQA, LoCoMo, 2WikiMultiHopQA, LongMemEval, ALFWorld, and WebShop.

Loads authentic benchmark files (including the bundled `data/hotpotqa/` and
`data/locomo/` datasets) into unified `Episode` streams with exact document
identifiers and character offsets for post-hoc controlled provenance P_q
(Appendix H).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ugpo.data.streaming import Episode, GoldSupportUnit, RequestItem, StreamUnit


_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read either a JSON list file or newline-delimited JSONL records."""
    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "data" in data:
        return list(data["data"])
    if isinstance(data, list):
        return data
    return [data]


def _normalize_hotpot_context(
    raw_context: Any,
) -> List[Tuple[str, List[str]]]:
    """Handle both HuggingFace parquet dict-of-lists and official JSON list-of-pairs formats."""
    if isinstance(raw_context, dict) and "title" in raw_context and "sentences" in raw_context:
        return [
            (str(title), [str(s) for s in sents])
            for title, sents in zip(raw_context["title"], raw_context["sentences"])
        ]
    if isinstance(raw_context, list):
        pairs: List[Tuple[str, List[str]]] = []
        for entry in raw_context:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                pairs.append((str(entry[0]), [str(s) for s in entry[1]]))
        return pairs
    return []


def _normalize_hotpot_supports(raw_supports: Any) -> List[Tuple[str, int]]:
    """Parse HotpotQA / 2Wiki supporting_facts into (title, sent_id) tuples."""
    if isinstance(raw_supports, dict) and "title" in raw_supports and "sent_id" in raw_supports:
        return [
            (str(title), int(sent_id))
            for title, sent_id in zip(raw_supports["title"], raw_supports["sent_id"])
        ]
    if isinstance(raw_supports, list):
        out: List[Tuple[str, int]] = []
        for entry in raw_supports:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                out.append((str(entry[0]), int(entry[1])))
        return out
    return []


def load_hotpotqa(
    path: str | Path = "data/hotpotqa/hotpot_dev_distractor.parquet",
    limit: Optional[int] = None,
    chunk_by_sentence: bool = False,
) -> List[Episode]:
    """Load HotpotQA into streaming long-document Episodes (Section 4.1).

    Supports both HuggingFace `.parquet` files (bundled in `data/hotpotqa/`)
    and official `.json`/`.jsonl` files. Documents arrive sequentially as
    `StreamUnit` chunks; supporting sentences are recorded strictly as
    post-hoc `GoldSupportUnit` annotations for controlled provenance P_q.
    """
    file_path = Path(path)
    if file_path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(file_path)
        n_rows = table.num_rows if limit is None else min(limit, table.num_rows)
        cols = table.column_names
        records = [
            {col: table[col][idx].as_py() for col in cols}
            for idx in range(n_rows)
        ]
    else:
        records = _read_json_or_jsonl(file_path)
        if limit is not None:
            records = records[:limit]

    episodes: List[Episode] = []
    for idx, row in enumerate(records):
        ep_id = str(row.get("id", row.get("_id", f"hotpot_{idx}")))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        q_type = str(row.get("type", "bridge"))
        level = str(row.get("level", "hard"))

        context_docs = _normalize_hotpot_context(row.get("context"))
        support_pairs = _normalize_hotpot_supports(row.get("supporting_facts"))

        stream_units: List[StreamUnit] = []
        sentence_offsets: Dict[Tuple[str, int], Tuple[int, int, str]] = {}
        total_words = 0

        for doc_idx, (title, sentences) in enumerate(context_docs):
            cursor = 0
            joined_parts: List[str] = []
            for sent_idx, sent in enumerate(sentences):
                clean_sent = sent.strip()
                start_offset = cursor
                end_offset = start_offset + len(clean_sent)
                sentence_offsets[(title, sent_idx)] = (start_offset, end_offset, clean_sent)
                cursor = end_offset + 1
                joined_parts.append(clean_sent)

                if chunk_by_sentence and clean_sent:
                    unit_text = f"{title}: {clean_sent}"
                    total_words += len(unit_text.split())
                    stream_units.append(
                        StreamUnit(
                            unit_id=f"{ep_id}_d{doc_idx}_s{sent_idx}",
                            text=unit_text,
                            session_id=doc_idx,
                            source_domain="wikipedia",
                            source_doc_id=title,
                            char_start=start_offset,
                            char_end=end_offset,
                            metadata={"title": title, "sent_id": sent_idx},
                        )
                    )

            if not chunk_by_sentence:
                doc_body = " ".join(joined_parts).strip()
                if doc_body:
                    unit_text = f"{title}: {doc_body}"
                    total_words += len(unit_text.split())
                    stream_units.append(
                        StreamUnit(
                            unit_id=f"{ep_id}_d{doc_idx}",
                            text=unit_text,
                            session_id=doc_idx,
                            source_domain="wikipedia",
                            source_doc_id=title,
                            char_start=0,
                            char_end=len(doc_body),
                            metadata={"title": title, "num_sentences": len(sentences)},
                        )
                    )

        gold_supports: List[GoldSupportUnit] = []
        for title, sent_idx in support_pairs:
            if (title, sent_idx) in sentence_offsets:
                c_start, c_end, s_text = sentence_offsets[(title, sent_idx)]
                gold_supports.append(
                    GoldSupportUnit(
                        source_doc_id=title,
                        char_start=c_start,
                        char_end=c_end,
                        unit_id=f"{title}#s{sent_idx}",
                        text=s_text,
                    )
                )

        req = RequestItem(
            request_id=f"{ep_id}_q0",
            question=question,
            answer=answer,
            request_type=q_type,
            session_id=len(stream_units),
            context_length=total_words,
            gold_supports=gold_supports,
            metadata={"level": level, "type": q_type},
        )
        episodes.append(
            Episode(
                episode_id=ep_id,
                benchmark="hotpotqa",
                stream_units=stream_units,
                requests=[req],
                metadata={"level": level, "type": q_type},
            )
        )
    return episodes


def load_2wikimultihopqa(
    path: str | Path,
    limit: Optional[int] = None,
    chunk_by_sentence: bool = False,
) -> List[Episode]:
    """Load 2WikiMultiHopQA into streaming long-document Episodes (Section 4.1)."""
    episodes = load_hotpotqa(path=path, limit=limit, chunk_by_sentence=chunk_by_sentence)
    for ep in episodes:
        ep.benchmark = "2wiki"
    return episodes


def load_locomo(
    path: str | Path = "data/locomo/locomo10.json",
    limit: Optional[int] = None,
) -> List[Episode]:
    """Load the official LoCoMo multi-session conversational memory dataset.

    Preserves exact session chronology (`session_1`, `session_2`, ...) and
    dialogue turn identifiers (`dia_id`, e.g. `D1:3`) so cross-session
    write-to-use delays and controlled provenance P_q align cleanly.
    """
    records = _read_json_or_jsonl(Path(path))
    if limit is not None:
        records = records[:limit]

    episodes: List[Episode] = []
    for idx, conv_record in enumerate(records):
        sample_id = str(conv_record.get("sample_id", f"locomo_{idx}"))
        conversation = conv_record.get("conversation", {})

        # Sort session keys numerically: session_1, session_2, ..., session_N.
        session_nums: List[int] = []
        for key in conversation.keys():
            match = _SESSION_KEY_RE.match(key)
            if match:
                session_nums.append(int(match.group(1)))
        session_nums.sort()

        stream_units: List[StreamUnit] = []
        turn_lookup: Dict[str, Tuple[int, int, str]] = {}
        total_words = 0

        for sess_num in session_nums:
            turns = conversation.get(f"session_{sess_num}", [])
            sess_time = str(conversation.get(f"session_{sess_num}_date_time", ""))
            cursor = 0
            for turn in turns:
                speaker = str(turn.get("speaker", "Speaker"))
                dia_id = str(turn.get("dia_id", f"D{sess_num}:{len(stream_units)}"))
                raw_text = str(turn.get("text", "")).strip()
                if not raw_text:
                    continue
                formatted = f"[{sess_time}] {speaker}: {raw_text}" if sess_time else f"{speaker}: {raw_text}"
                c_start = cursor
                c_end = c_start + len(formatted)
                cursor = c_end + 1
                total_words += len(formatted.split())
                turn_lookup[dia_id] = (c_start, c_end, formatted)

                stream_units.append(
                    StreamUnit(
                        unit_id=dia_id,
                        text=formatted,
                        session_id=sess_num,
                        source_domain="dialogue",
                        source_doc_id=dia_id,
                        char_start=c_start,
                        char_end=c_end,
                        timestamp=sess_time,
                        metadata={"speaker": speaker, "dia_id": dia_id, "session": sess_num},
                    )
                )

        max_session = session_nums[-1] if session_nums else 1
        requests: List[RequestItem] = []
        for q_idx, qa in enumerate(conv_record.get("qa", [])):
            category = int(qa.get("category", 1))
            question = str(qa.get("question", ""))
            # Category 5 in official LoCoMo stores adversarial unanswerable QA pairs.
            if "answer" in qa and qa["answer"] is not None:
                answer = str(qa["answer"])
            else:
                answer = str(qa.get("adversarial_answer", ""))

            evidence_ids = qa.get("evidence", []) or []
            gold_supports: List[GoldSupportUnit] = []
            for ev_id in evidence_ids:
                ev_str = str(ev_id).strip()
                if ev_str in turn_lookup:
                    c_start, c_end, turn_text = turn_lookup[ev_str]
                    gold_supports.append(
                        GoldSupportUnit(
                            source_doc_id=ev_str,
                            char_start=c_start,
                            char_end=c_end,
                            unit_id=ev_str,
                            text=turn_text,
                        )
                    )
                else:
                    gold_supports.append(
                        GoldSupportUnit(source_doc_id=ev_str, unit_id=ev_str)
                    )

            requests.append(
                RequestItem(
                    request_id=f"{sample_id}_q{q_idx}",
                    question=question,
                    answer=answer,
                    request_type=f"category_{category}",
                    session_id=max_session + 1,
                    context_length=total_words,
                    gold_supports=gold_supports,
                    metadata={
                        "category": category,
                        "evidence_ids": [str(e) for e in evidence_ids],
                        "is_adversarial": category == 5,
                    },
                )
            )

        episodes.append(
            Episode(
                episode_id=sample_id,
                benchmark="locomo",
                stream_units=stream_units,
                requests=requests,
                metadata={
                    "num_sessions": len(session_nums),
                    "speaker_a": conversation.get("speaker_a", ""),
                    "speaker_b": conversation.get("speaker_b", ""),
                },
            )
        )
    return episodes


def load_longmemeval(
    path: str | Path,
    limit: Optional[int] = None,
) -> List[Episode]:
    """Load LongMemEval multi-session interactive memory benchmark records."""
    records = _read_json_or_jsonl(Path(path))
    if limit is not None:
        records = records[:limit]

    episodes: List[Episode] = []
    for idx, row in enumerate(records):
        q_id = str(row.get("question_id", f"longmemeval_{idx}"))
        q_type = str(row.get("question_type", "single-session-user"))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        haystack_sessions = row.get("haystack_sessions", []) or []
        haystack_dates = row.get("haystack_dates", []) or []
        haystack_session_ids = row.get("haystack_session_ids", []) or []
        answer_session_ids = set(str(s) for s in (row.get("answer_session_ids", []) or []))

        stream_units: List[StreamUnit] = []
        gold_supports: List[GoldSupportUnit] = []
        total_words = 0

        for s_idx, turns in enumerate(haystack_sessions):
            sess_id_str = str(haystack_session_ids[s_idx]) if s_idx < len(haystack_session_ids) else f"sess_{s_idx}"
            sess_date = str(haystack_dates[s_idx]) if s_idx < len(haystack_dates) else ""
            cursor = 0
            for t_idx, turn in enumerate(turns):
                role = str(turn.get("role", "user"))
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                formatted = f"[{sess_date}] {role}: {content}" if sess_date else f"{role}: {content}"
                c_start = cursor
                c_end = c_start + len(formatted)
                cursor = c_end + 1
                total_words += len(formatted.split())
                unit_id = f"{sess_id_str}_t{t_idx}"

                stream_units.append(
                    StreamUnit(
                        unit_id=unit_id,
                        text=formatted,
                        session_id=s_idx + 1,
                        source_domain="assistant_chat",
                        source_doc_id=sess_id_str,
                        char_start=c_start,
                        char_end=c_end,
                        timestamp=sess_date,
                        metadata={"role": role, "session_id_str": sess_id_str},
                    )
                )
                if sess_id_str in answer_session_ids or bool(turn.get("has_answer", False)):
                    gold_supports.append(
                        GoldSupportUnit(
                            source_doc_id=sess_id_str,
                            char_start=c_start,
                            char_end=c_end,
                            unit_id=unit_id,
                            text=formatted,
                        )
                    )

        req = RequestItem(
            request_id=f"{q_id}_q0",
            question=question,
            answer=answer,
            request_type=q_type,
            session_id=len(haystack_sessions) + 1,
            context_length=total_words,
            gold_supports=gold_supports,
            metadata={
                "question_type": q_type,
                "question_date": row.get("question_date", ""),
            },
        )
        episodes.append(
            Episode(
                episode_id=q_id,
                benchmark="longmemeval",
                stream_units=stream_units,
                requests=[req],
            )
        )
    return episodes


def load_agentic_tasks(
    path: str | Path,
    benchmark: str = "alfworld",
    limit: Optional[int] = None,
) -> List[Episode]:
    """Load ALFWorld or WebShop agentic transfer task definitions (Appendix H)."""
    records = _read_json_or_jsonl(Path(path))
    if limit is not None:
        records = records[:limit]

    episodes: List[Episode] = []
    for idx, row in enumerate(records):
        ep_id = str(row.get("task_id", row.get("id", f"{benchmark}_{idx}")))
        goal = str(row.get("goal", row.get("instruction", "")))
        observations = row.get("observations", []) or []
        stream_units: List[StreamUnit] = []
        total_words = 0

        for step_idx, obs in enumerate(observations):
            obs_text = str(obs.get("text", obs) if isinstance(obs, dict) else obs).strip()
            if not obs_text:
                continue
            total_words += len(obs_text.split())
            doc_id = str(obs.get("doc_id", f"{ep_id}_obs_{step_idx}")) if isinstance(obs, dict) else f"{ep_id}_obs_{step_idx}"
            stream_units.append(
                StreamUnit(
                    unit_id=doc_id,
                    text=obs_text,
                    session_id=step_idx,
                    source_domain=benchmark,
                    source_doc_id=doc_id,
                    char_start=0,
                    char_end=len(obs_text),
                )
            )

        gold_supports = [
            GoldSupportUnit(
                source_doc_id=str(g.get("source_doc_id", g)) if isinstance(g, dict) else str(g),
                text=str(g.get("text", "")) if isinstance(g, dict) else "",
            )
            for g in (row.get("gold_objects", row.get("gold_attributes", [])) or [])
        ]
        req = RequestItem(
            request_id=f"{ep_id}_goal",
            question=goal,
            answer=str(row.get("target_action", row.get("target_asin", ""))),
            request_type=str(row.get("task_type", benchmark)),
            session_id=len(stream_units),
            context_length=total_words,
            gold_supports=gold_supports,
            metadata={"benchmark": benchmark},
        )
        episodes.append(
            Episode(
                episode_id=ep_id,
                benchmark=benchmark,
                stream_units=stream_units,
                requests=[req],
            )
        )
    return episodes


def load_benchmark(
    benchmark: str,
    path: str | Path,
    limit: Optional[int] = None,
) -> List[Episode]:
    """Unified benchmark loader dispatching by canonical dataset name."""
    name = benchmark.lower().strip()
    if name == "hotpotqa":
        return load_hotpotqa(path=path, limit=limit)
    if name in ("2wiki", "2wikimultihopqa"):
        return load_2wikimultihopqa(path=path, limit=limit)
    if name == "locomo":
        return load_locomo(path=path, limit=limit)
    if name == "longmemeval":
        return load_longmemeval(path=path, limit=limit)
    if name in ("alfworld", "webshop"):
        return load_agentic_tasks(path=path, benchmark=name, limit=limit)
    raise ValueError(f"Unknown benchmark '{benchmark}'. Expected hotpotqa, 2wiki, locomo, longmemeval, alfworld, or webshop.")
