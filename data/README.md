# Datasets

This repository includes two complete official benchmark datasets directly under `data/`, covering both primary long-context memory regimes evaluated in the paper (Section 4.1 and Appendix H):

| Benchmark | Regime | Bundled File | Size | Examples | Gold Provenance Units |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **HotpotQA** (Yang et al., 2018) | Streaming long-document multi-hop QA | `data/hotpotqa/hotpot_dev_distractor.parquet` | 27 MB | 7,405 questions (distractor dev) | `supporting_facts` (`title`, `sent_id`) |
| **LoCoMo** (Maharana et al., 2024) | Multi-session conversational memory | `data/locomo/locomo10.json` | 2.7 MB | 10 multi-session dialogues, 1,986 QAs | `evidence` turn IDs (`dia_id`, e.g., `D1:3`) |

> **Note on Controlled Provenance ($P_q$):** Gold supporting annotations are strictly reserved for post-hoc attribution precision and recall evaluation (`src/ugpo/eval/provenance.py`, Appendix H) and are never passed into UGPO training, grouping, or reward computation.

## Bundled Dataset Schemas

### 1. HotpotQA (`data/hotpotqa/hotpot_dev_distractor.parquet`)
Source: `hotpotqa/hotpot_qa` (`distractor`, `validation` split).  
Each record contains:
- `id`: unique question identifier
- `question`, `answer`: multi-hop question and gold answer string
- `type`: reasoning type (`bridge` or `comparison`)
- `level`: difficulty annotation
- `context`: parallel lists of document `title` and paragraph `sentences`, streamed sequentially as `StreamUnit` chunks in `load_hotpotqa()`
- `supporting_facts`: gold `(title, sent_id)` pairs aligned by character offset during post-hoc provenance evaluation

### 2. LoCoMo (`data/locomo/locomo10.json`)
Source: `snap-research/locomo` (`data/locomo10.json`).  
Each conversation contains:
- `sample_id`: conversation identifier
- `conversation`: chronologically ordered `session_1 ... session_N` turn lists with `session_k_date_time`, `speaker`, `dia_id`, and `text`
- `qa`: multi-session questions spanning all five official categories (1: single-hop, 2: temporal, 3: open-domain, 4: multi-hop, 5: adversarial unanswerable) with `evidence` turn identifiers

## Additional Benchmarks Supported by `src/ugpo/data/loaders.py`

The unified loader (`load_benchmark`) also supports the remaining benchmarks evaluated in the paper once downloaded locally:
- **2WikiMultiHopQA** (Ho et al., 2020): pass `--benchmark 2wiki --data-path <path/to/2wiki.json>`
- **LongMemEval** (Wu et al., 2025): pass `--benchmark longmemeval --data-path <path/to/longmemeval_s_cleaned.json>`
- **ALFWorld** (Shridhar et al., 2021) & **WebShop** (Yao et al., 2023a): pass `--benchmark alfworld|webshop --data-path <path/to/tasks.jsonl>`
