# Medical RAG: Baseline vs Fusion Comparison

A thesis project implementing and evaluating two Retrieval-Augmented Generation (RAG) architectures—**Baseline RAG** and **RAG-Fusion**—for Indonesian medical document analysis. The system processes patient records, clinical notes, and nutritional assessments to answer complex clinical queries with high factual accuracy.

---

## Overview

This project investigates whether **RAG-Fusion** (multi-query expansion + Reciprocal Rank Fusion) improves retrieval quality over standard dense retrieval when answering clinical questions from Indonesian medical records. The pipeline handles the full lifecycle: PDF parsing → text cleaning → metadata enrichment → vector indexing → synthetic evaluation dataset generation → comparative evaluation.

### Key Features

- **Two RAG architectures** implemented and benchmarked side-by-side
- **Local LLM inference** with Qwen3-4B-Instruct (4-bit quantized via BitsAndBytes)
- **Medical-domain optimizations**: diagnosis-aware filtering, source-tagged contexts, hallucination-minimizing prompts
- **Synthetic evaluation dataset** generated via multi-hop QA synthesis (GPT-4o-mini)
- **Comprehensive evaluation** using DeepEval (Faithfulness, Answer Relevancy) and programmatic retrieval metrics (MRR@K, Contextual Precision/Recall)

---

## Architecture Comparison

| Component | Baseline RAG | RAG-Fusion |
|-----------|--------------|------------|
| **Query Processing** | Single dense retrieval with instruction prefix | Multi-query expansion (1 original + 3 generated) |
| **Retrieval** | Top-K cosine similarity search | Parallel multi-retrieval (2K per query) |
| **Ranking** | Native vector scores | Reciprocal Rank Fusion (k=60) |
| **Generation** | Qwen3-4B-Instruct (local, 4-bit) | Same LLM, fused context |
| **Metadata Filtering** | ✅ Diagnosis-based | ✅ Diagnosis-based |

### RAG-Fusion Pipeline

```text
User Query
    │
    ├─► Query Generator (GPT-4o-mini) ──► 3 expanded queries
    │
    ├─► Original Query + 3 Expanded → Parallel Dense Retrieval (Qdrant)
    │       │                           │
    │       ▼                           ▼
    │   Top 6 docs                  Top 6 docs (×3)
    │       │                           │
    └───────► Reciprocal Rank Fusion (k=60)
                    │
                    ▼
            Top-K Fused Documents
                    │
                    ▼
         Qwen3-4B Generation (Local)
```

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **LLM (Generator)** | Qwen/Qwen3-4B-Instruct-2507 (local, 4-bit NF4 quantization) |
| **LLM (Query Expansion / Judge)** | GPT-4o-mini (OpenAI API) |
| **Embeddings** | intfloat/multilingual-e5-large (or configurable via HuggingFace) |
| **Vector Database** | Qdrant (Cosine distance, payload indexing for metadata filtering) |
| **Orchestration** | LangChain (LCEL chains, prompt templates) |
| **Document Parsing** | Docling (layout-aware PDF → Markdown) |
| **Evaluation** | DeepEval (Faithfulness, Answer Relevancy), programmatic MRR@K, Contextual Precision/Recall |
| **Experiment Tracking** | LangSmith |
| **Environment** | Python 3.10+, CUDA, PyTorch, Transformers, BitsAndBytes |

---

## Project Structure

```text
playground/
├── main.py                      # Entry point: runs both RAG systems on golden dataset
├── rag_pipeline.py              # Core RAG implementations (Baseline + Fusion)
├── indexing_medical_v3.py       # Vector indexing pipeline (Qdrant upload)
├── pdf_to_md_parser.py          # PDF → Markdown conversion (Docling)
├── text_cleaning.py             # Noise removal from parsed Markdown
├── synthetic_data_generator.py  # Golden dataset synthesis (multi-hop QA)
├── config/
│   ├── __init__.py
│   └── config.py                # Centralized configuration (env-driven)
├── knowledge-base/              # Raw, processed, and indexed data
├── notebooks/                   # Exploration & evaluation notebooks
├── notes/                       # Development notes
├── research/                    # Experimental variants
├── requirements.txt
└── skripsi.md                   # Full thesis documentation (Indonesian)
```

---

## Configuration

Key settings in `config/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model identifier |
| `EMBEDDING_DIM` | `1536` | Embedding dimension |
| `MODEL_ID` | `Qwen/Qwen3-4B-Instruct-2507` | Local generator model |
| `COLLECTION_NAME` | `2026_03_16__1147_enriched_medical_reports` | Qdrant collection |
| `TOP_K` | `3` | Documents retrieved per query |
| `TEMPERATURE` | `0.1` | Generation temperature |
| `MAX_NEW_TOKENS` | `512` | Max tokens per generation |

---

## Evaluation Methodology

### Retrieval Metrics (Programmatic)

- **MRR@K**: Mean Reciprocal Rank of first relevant document
- **Contextual Precision**: Relevance ranking quality (threshold=0.5)
- **Contextual Recall**: Coverage of ground-truth information (threshold=0.5)

### Generation Metrics (DeepEval / LLM-as-a-Judge)

- **Faithfulness**: Factual consistency with retrieved context (threshold=0.5)
- **Answer Relevancy**: Directness and topical alignment to query (threshold=0.5)

Evaluation runs in `notebooks/evaluation_deepeval.ipynb` with `AsyncConfig(max_concurrent=3, throttle_value=2)`.

---

## Medical Domain Adaptations

1. **Diagnosis-aware filtering**: Queries filtered by `metadata.medical_diagnosis` to constrain search to same-patient records
2. **Source-tagged contexts**: Each retrieved chunk labeled with source document to prevent cross-patient contamination
3. **Clinical prompt constraints**: System prompt enforces medical assistant persona, prohibits hallucination, mandates "Informasi tidak lengkap." when context insufficient
4. **Indonesian medical terminology**: Query expansion preserves clinical identifiers (diagnosis, age, gender) while expanding abbreviations (e.g., BBLR → Berat Badan Lahir Rendah)

---

### Comparative Metrics

Evaluation results on the golden dataset (50 multi-hop samples):

| Category | Metric | RAG Baseline | RAG-Fusion | Improvement |
|----------|--------|--------------|------------|-------------|
| Retrieval | MRR@3 | 0.3967 | 0.4700 | +7.33% |
| Retrieval | Contextual Precision | 0.6767 | 0.7933 | +11.66% |
| Retrieval | Contextual Recall | 0.8280 | 0.8827 | +5.47% |
| Generation | Faithfulness | 0.9460 | 0.9511 | +0.51% |
| Generation | Answer Relevancy | 0.8273 | 0.8307 | +0.34% |

> RAG-Fusion shows measurable improvement across all metrics, with the largest gains in **Contextual Precision (+11.66%)** and **MRR@3 (+7.33%)**, indicating that multi-query expansion + Reciprocal Rank Fusion significantly improves the ranking quality of retrieved medical documents. Generation metrics show smaller but consistent gains, suggesting both systems produce grounded and relevant answers, with RAG-Fusion having a slight edge.

---

## References

- [RAG-Fusion: Reciprocal Rank Fusion for Multi-Query Retrieval](https://arxiv.org/abs/2402.03367)
- [DeepEval: LLM Evaluation Framework](https://github.com/confident-ai/deepeval)
- [Qdrant Vector Database](https://qdrant.tech/)
- [Docling: Document Understanding](https://github.com/DS4SD/docling)
- [LangChain Expression Language](https://python.langchain.com/docs/expression_language/)
