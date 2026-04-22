# Amazon Books Retrieval System
## Dual-Method Information Retrieval with RAG Integration

A retrieval system combining **BM25 keyword-based search**, **semantic embedding-based search**, and **Retrieval-Augmented Generation (RAG)** to find relevant books from the Amazon Reviews 2023 dataset and generate AI-powered answers using Groq LLM.

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Status Active](https://img.shields.io/badge/Status-Active-success)
![Final Submission](https://img.shields.io/badge/Final-Submission-blueviolet)

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Running the System](#running-the-system)
- [Scale and Runtime](#scale-and-runtime)
- [Testing the System](#testing-the-system)
- [How the System Works](#how-the-system-works)
- [Retrieval Methods](#retrieval-methods)
- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
- [Groq LLM Integration](#groq-llm-integration)
- [Project Structure](#project-structure)
- [Data Files and Field Selection](#data-files-and-field-selection)
- [Text Preprocessing](#text-preprocessing)
- [Evaluation Results](#evaluation-results)
- [Team](#team)
- [References and Resources](#references-and-resources)
- [License](#license)

---

## Overview

### Project Goal

Building a retrieval system teaches us what keyword matching and semantic understanding can achieve independently. Users search in different ways: some use exact keywords ("mystery novel"), others describe intent ("book to help with anxiety"), and some have complex needs ("best book to learn machine learning with no math background"). This project explores how BM25 and semantic search handle these different patterns, combines them in hybrid mode, and then uses Retrieval-Augmented Generation to intelligently answer user queries using retrieved books as context.

### Dataset: Amazon Reviews 2023

We use the **Amazon Reviews 2023** dataset from UC San Diego's McAuley Lab, containing 571 million+ reviews across 34 product categories collected from May 1996 through September 2023.

**Why "Books" Category:** Book reviews are detailed and substantive (users write paragraphs explaining what they liked), metadata is well-structured (clear titles, authors, descriptions), the dataset is large but manageable (11.7M reviews, 3.1M books), search queries are naturally diverse (by genre, topic, author, style, learning intent), and results are easy to verify (humans know what books are about).

---

## Getting Started

### Requirements
- Python 3.11+
- 16GB RAM
- 10GB disk space
- Groq API key (free tier)

### Setup and Installation

```bash
# Clone repository
git clone https://github.com/UBC-MDS/DSCI_575_project_jchuang_esteki.git
cd DSCI_575_project_jchuang_esteki

# Create and activate environment
conda env create -f environment.yml
conda activate 575-project

# Create .env file with Groq API key
cp env.example .env
# Edit .env: add your GROQ_API_KEY=gsk_...
```

---

## Running the System

**First time (generate indexes and RAG pipeline):**
```bash
jupyter notebook

# Run notebooks in order:
# 01_exploration
# 02_data_preparation
# 03_bm25_keyword_search
# 04_semantic_embedding_search
# 05_evaluation_and_verification
# 06_hybrid_retrieval
# 07_rag_pipeline
# 08_llm_comparison
```

**After setup:**
```bash
streamlit run app/app.py
# Opens at http://localhost:8501
```

---

## Scale and Runtime

The final submission scales the pipeline to 100,000 sampled reviews (Option 3 of the final milestone). After preprocessing (dropping reviews under 20 characters) the retrieval corpus contains **91,850 enriched documents**.

### Sampling Strategy (Two-tier)

`notebooks/02_data_preparation.ipynb` samples directly from the compressed raw file with DuckDB, never loading the full 11.7M reviews into RAM:

1. **Stratified first.** The notebook tries a stratified SQL query that allocates rows proportionally across the 1 to 5 rating buckets.
2. **Random fallback.** If the stratified query fails on the 11.7M row stream (its `OVER / RANDOM` plan can exhaust DuckDB's working memory), the notebook automatically falls back to `ORDER BY RANDOM() LIMIT 100000`. This is the path that ran on the final 100K build. The resulting rating distribution still matches the natural skew of Amazon book reviews.

### Expected Runtime on a 16 GB laptop

| Stage | Approximate time |
|---|---|
| Data preparation (notebook 02) | 3 to 5 minutes |
| BM25 index build (notebook 03) | 2 to 4 minutes |
| FAISS index build (notebook 04, `batch_size=128`) | 5 to 10 minutes |
| Evaluation and RAG notebooks (05 to 08) | 5 to 10 minutes |
| **Total end-to-end rebuild** | **15 to 25 minutes** |

### Disk Footprint of Generated Artifacts

| File | Size |
|---|---|
| `data/processed/corpus.pkl` | ~51 MB |
| `data/processed/books_sample.parquet` | ~30 MB |
| `data/processed/bm25_index.pkl` | ~150 MB |
| `data/processed/semantic_index/faiss_index` | ~140 MB |

---

## Testing the System

Try these example queries with the app running:

#### Easy Queries (BM25 works well):
```
  mystery novel
  cookbook recipes
  science fiction space
```

#### Medium Queries (Semantic works well):
```
  book to help with anxiety
  guide for first time parents
  story about finding yourself
```

#### Complex Queries (Hybrid and RAG work better):
```
  best book to learn machine learning with no math background
  historical fiction set in world war 2 from a female perspective
  self help book for overcoming procrastination and building better habits
```

**How to test:** Paste a query, try BM25 (fast, exact matches), Semantic (slower, understands intent), Hybrid (adjustable weights), or RAG (AI-generated answer with retrieved books). Compare results and see which method works better for different query types.

For detailed evaluation results, see `results/milestone1_discussion.md`, `results/milestone2_discussion.md`, and `results/final_discussion.md`.

---

## How the System Works

### Data Processing Pipeline

Raw data transforms through the following notebooks, each producing outputs for the next stage:

- **Notebook 01: Exploration** explores the dataset structure by loading sample records, inspecting fields and distributions, visualizing rating patterns, and documenting field selection rationale.

- **Notebook 02: Data Preparation** processes the 11.7M Books reviews into a retrieval-ready corpus. It uses DuckDB to stream the compressed file without memory overflow, samples 100,000 reviews using a stratified query with a random-sample fallback (see the Scale and Runtime section), joins reviews with product metadata on `parent_asin`, applies consistent text preprocessing, and concatenates product title with review text into unified documents. Outputs: `corpus.pkl` (91,850 preprocessed documents) and `books_sample.parquet`.

- **Notebook 03: BM25 Indexing** builds keyword-based search capability by loading the corpus, tokenizing documents consistently, building an inverted index using `rank_bm25`, and testing on sample queries. Output: `bm25_index.pkl`.

- **Notebook 04: Semantic Indexing** builds meaning-based search capability by loading the corpus, encoding all documents to 384-dimensional vectors using `all-MiniLM-L6-v2` sentence-transformers model with `batch_size=128`, building a FAISS index for fast similarity search, and testing on sample queries. Output: `semantic_index/` directory.

- **Notebook 05: Evaluation** compares both methods on a diverse set of queries, creates retrieval results, analyzes performance patterns, and documents findings. Output: `results/milestone1_discussion.md`.

- **Notebook 06: Hybrid Retrieval** combines BM25 and semantic search with adjustable weights, tests on sample queries, and provides users control to emphasize either keyword matching or semantic understanding.

- **Notebook 07: RAG Pipeline** auto-generates RAG modules (chunking, prompts, pipeline) and tests integration with the retrieval system. Output: `src/chunking.py`, `src/prompts.py`, `src/rag_pipeline.py`.

- **Notebook 08: LLM Comparison** runs the same five queries through LLaMA 3.3 70B and LLaMA 3.1 8B (via Groq) using identical retrieved context and the same `RECOMMENDATION` prompt. Results and the selected default are documented in `results/final_discussion.md`.

---

## Retrieval Methods

- **BM25 Keyword Search (Notebook 03):** Indexes every word and which documents contain it. When searching "mystery novel", finds documents with both words and scores based on term frequency and inverse document frequency. Speed: <10ms per query. Strengths: fast, transparent, exact keyword matching. Limitations: no synonyms, words treated independently, no intent understanding.

- **Semantic Search (Notebook 04):** Converts documents and queries into vectors where similar meaning corresponds to nearby locations. The pre-trained model learned that "parent", "parenting", "newborn", "baby" should be close in vector space. When searching "guide for first time parents", finds the query embedding's nearest neighbors and returns results ranked by distance. Speed: ~100ms per query. Strengths: understands intent, handles synonyms, context-aware. Limitations: slower, less transparent, may miss exact matches.

- **Hybrid Search (Notebook 06):** Combines both methods with adjustable weights (0-100% BM25, rest semantic), giving users control to emphasize either keyword matching or semantic understanding. Speed: ~150ms per query. Strengths: balances precision and recall, flexible for different query types. Optimal for mixed queries combining keywords and intent.

---

## Retrieval-Augmented Generation (RAG)

RAG combines retrieval with generative AI to answer questions using book data as context:

```
User Query
    ↓
Retrieve Top-K Books (Hybrid Search)
    ↓
Build Context from Retrieved Reviews
    ↓
Send to Groq LLM with Prompt Template
    ↓
AI-Generated Answer with Sources
    ↓
Display Answer + Retrieved Books
```

### RAG Workflow

1. **Retrieve:** Hybrid search finds top-5 most relevant books
2. **Context:** Concatenates book titles and reviews into context window
3. **Prompt:** Formats as "Based on these reviews, answer: [question]"
4. **Generate:** Groq LLM produces intelligent response
5. **Display:** Shows answer and which books were used

### RAG Components

Generated automatically by Notebook 07:

- **`src/chunking.py`** - DocumentChunker: Splits long documents into 500-char chunks with 50-char overlap
- **`src/prompts.py`** - RAGPrompts: Balanced vs Strict prompt templates for different answer styles
- **`src/rag_pipeline.py`** - RAGPipeline: Orchestrates retrieval, context building, and generation

---

## Groq LLM Integration

Groq provides free, fast LLM access for RAG generation.

### Getting Started with Groq

1. **Sign up (free, no credit card):** https://console.groq.com
2. **Create API key:** Copy the key starting with `gsk_`
3. **Add to project:**
   ```bash
   cp env.example .env
   # Edit .env: GROQ_API_KEY=gsk_your_key_here
   ```
4. **Use in app:** Select "Groq (Production)" in sidebar

### Model Selection

The LLM comparison in `notebooks/08_llm_comparison.ipynb` tested **LLaMA 3.3 70B** (`llama-3.3-70b-versatile`) against **LLaMA 3.1 8B** (`llama-3.1-8b-instant`) on the five project queries. The 70B model produced more grounded, complete, and hallucination-free responses, and is documented in `results/final_discussion.md` as the preferred default.

### Runtime Fallback Chain

The Streamlit app (`app/app.py`) wraps the LLM call in an auto-fallback chain so a deprecated or rate-limited model does not break the app:

1. `llama-3.2-90b-vision-preview`
2. `llama-3.1-70b-versatile`
3. `llama-3.1-8b-instant`
4. `gemma2-9b-it`
5. `mixtral-8x7b-32768`

If a model returns a `decommissioned` or `not supported` error, the app silently cascades to the next one. This keeps the app resilient to Groq's ongoing model lifecycle without requiring manual config changes.

---

## Project Structure

```
DSCI_575_project_jchuang_esteki/
│
├── README.md                         # Project documentation
├── environment.yml                   # Conda dependencies
├── env.example                       # Environment variables template
├── .gitignore                        # Excludes data and credentials
│
├── data/
│   ├── raw/                          # Raw dataset (NOT in git)
│   │   ├── Books.jsonl.gz            # Reviews: 11.7M records
│   │   └── meta_Books.jsonl.gz       # Metadata: 3.1M records
│   └── processed/                    # Generated by notebooks (NOT in git)
│       ├── books_sample.parquet      # 91,850 sampled reviews from 02
│       ├── corpus.pkl                # 91,850 preprocessed documents from 02
│       ├── bm25_index.pkl            # BM25 index from 03
│       └── semantic_index/           # FAISS index from 04
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   │   └─ Load and inspect dataset; justify field selection
│   │
│   ├── 02_data_preparation.ipynb
│   │   └─ DuckDB stream → Stratified sample (random fallback) → Join → Preprocess → Combine
│   │
│   ├── 03_bm25_keyword_search.ipynb
│   │   └─ Tokenize → Build inverted index → Test queries
│   │
│   ├── 04_semantic_embedding_search.ipynb
│   │   └─ Encode documents (batch_size=128) → Build FAISS index → Test queries
│   │
│   ├── 05_evaluation_and_verification.ipynb
│   │   └─ Load indexes → Retrieve top-5 for 10 queries → Analyze results
│   │
│   ├── 06_hybrid_retrieval.ipynb
│   │   └─ Combine BM25 + Semantic → Adjustable weights → Test queries
│   │
│   ├── 07_rag_pipeline.ipynb
│   │   └─ Auto-generate RAG modules → Test integration
│   │
│   └── 08_llm_comparison.ipynb
│       └─ LLaMA 3.3 70B vs LLaMA 3.1 8B on 5 queries with identical context
│
├── src/
│   ├── bm25.py                       # BM25Retriever class
│   ├── semantic.py                   # SemanticRetriever class (batch_size=128)
│   ├── semantic_retriever.py         # SemanticRetriever alt implementation
│   ├── chunking.py                   # DocumentChunker (auto-generated by 07)
│   ├── prompts.py                    # RAGPrompts (auto-generated by 07)
│   ├── rag_pipeline.py               # RAGPipeline (auto-generated by 07)
│   ├── utils.py                      # preprocess_text(), tokenize(), load functions
│   └── hybrid.py                     # HybridRetriever class
│
├── results/
│   ├── milestone1_discussion.md      # Evaluation: 10 queries, BM25 vs Semantic
│   ├── milestone2_discussion.md      # Evaluation: Hybrid + RAG performance
│   ├── final_discussion.md           # Scaling, LLM experiment, Cloud deployment plan
│   └── rag_test_results.json         # RAG test results
│
├── app/
│   └── app.py                        # Streamlit web interface with 4 search methods + RAG
│
└── .env                              # Environment variables (NEVER commit)
    └── GROQ_API_KEY=...
```

### File Connections

```
           Raw Data (Books.jsonl.gz + meta_Books.jsonl.gz)
                                  ↓
                          Notebook 01 (Explore)
                                  ↓
 Notebook 02 (DuckDB stream → 100K sample w/ fallback → Join → Preprocess → Combine)
                                  ↓
                      corpus.pkl (91,850 documents)
                                  ↓
        ┌─────────────────────────┼─────────────────────────┐
        ↓                         ↓                         ↓
    Notebook 03              Notebook 04               src/utils.py
   (BM25 Index)            (Semantic Index)           (Preprocessing)
        ↓                         ↓
  bm25_index.pkl          semantic_index/
        └─────────────────────────┬─────────────────────────┘
                                  ↓
                        Notebook 05 (Evaluate)
                                  ↓
                     milestone1_discussion.md
                                  ↓
                        Notebook 06 (Hybrid)
                                  ↓
                   Combines BM25 + Semantic with weights
                                  ↓
                        Notebook 07 (RAG)
                                  ↓
              src/chunking.py, src/prompts.py, src/rag_pipeline.py
                                  ↓
                        Notebook 08 (LLM comparison)
                                  ↓
                     final_discussion.md (LLM choice)
                                  ↓
                             app/app.py
              (BM25 + Semantic + Hybrid + RAG Search Interface)
                                  ↓
                        .env (GROQ_API_KEY)
                                  ↓
                          Groq LLM API
                                  ↓
                        User Results with Answers
```

---

## Data Files and Field Selection

### Data Files

The project uses two primary files from the Books category:

**Reviews File: `Books.jsonl.gz`** - Contains 11.7 million user-written reviews. Each line is a JSON object with fields:

```
Books.jsonl.gz (Reviews File)
├── rating              [1-5 stars] User rating
├── title               Review headline/summary
├── text                Full review body
├── timestamp           When review was written
├── verified_purchase   Boolean: verified purchase or not
├── helpful_vote        Number of helpful votes
└── parent_asin         [KEY] Product identifier (links to metadata)
```

**Metadata File: `meta_Books.jsonl.gz`** - Contains 3.1 million product records. Each product has:

```
meta_Books.jsonl.gz (Metadata File)
├── asin                Unique product identifier
├── parent_asin         [KEY] Parent product (for variants)
├── title               Product title (book name)
├── description         Product description
├── price               Current price (may be null)
├── images              Product images
├── features            Key features list
├── main_category       Primary category (Books)
├── average_rating      Mean user rating (0-5)
└── store               Seller information
```

**Key Insight:** Reviews and metadata are separate files but linked via `parent_asin`. This allows combining product information ("The Da Vinci Code by Dan Brown") with user opinions ("this mystery novel kept me guessing until the end").

### Field Selection Rationale

For corpus building, we selected:

- **From Reviews:** `text` (full review body) + `rating` (5-star scale) + `verified_purchase` (credibility signal) + `helpful_vote` (community validation)
- **From Metadata:** `title` (book name) + `average_rating` (product rating distinct from review)

**Why these fields?** The review `text` contains rich semantic content for retrieval. The `rating` and `helpful_vote` provide relevance signals. The product `title` and `average_rating` give context about the book itself. Together, they create documents with both detailed content and metadata context, enabling both keyword and semantic search to work effectively.

---

## Text Preprocessing

All text is processed using consistent helper functions from `src/utils.py`.

### Processing Flow
```
Raw Text
   ↓
Lowercase
   "The Da Vinci Code"
→  "the da vinci code"
   ↓
Remove Punctuation
   "don't", "book?"
→  "dont", "book"
   ↓
Normalize Whitespace
   "this   is   text"
→  "this is text"
   ↓
Filter Noise
   Remove documents < 20 characters
```

Please ensure that all documents and queries are processed identically, as retrieval will fail otherwise.

The same preprocessing pipeline is used:
- During indexing (Notebooks 03 and 04)
- During query processing (Notebook 05 and `app.py` at runtime)

---

## Evaluation Results

### Milestone 1: Retrieval Methods

Notebook 05 evaluates BM25 vs Semantic Search across 10 queries:

* **Easy** (e.g., "mystery novel"): tests keyword matching
* **Medium** (e.g., "book to help with anxiety"): requires semantic understanding
* **Complex** (e.g., "best book to learn machine learning with no math background"): involves multiple constraints

### Key Findings

* **BM25:** Strong on exact keyword matching, struggles with ambiguous terms
* **Semantic:** Better at capturing user intent, handles multi-word concepts well
* **Both:** Struggle with vague queries, fail when relevant items don't exist

**Full results:** `results/milestone1_discussion.md`

### Milestone 2: Hybrid Search and RAG

Notebook 06 combines retrieval methods. Notebook 07 adds AI generation:

* **Hybrid Search:** Adjustable BM25/Semantic weighting achieves 15-20% better precision than either alone on complex queries
* **RAG:** Generates coherent answers by using top-5 books as context, works well for subjective queries ("why should I read this book?")

**Full results:** `results/milestone2_discussion.md` and `results/rag_test_results.json`

### Final Submission: Scale and LLM Experiment

Notebook 08 compares LLaMA 3.3 70B vs LLaMA 3.1 8B on 5 queries. The full dataset-scaling story, LLM comparison, cloud-deployment plan, and code-quality cleanups are in `results/final_discussion.md`.

* **Scale:** Pipeline now processes 100,000 sampled reviews (91,850 docs after preprocessing)
* **LLM:** LLaMA 3.3 70B chosen as preferred model based on five-query evaluation
* **Deployment:** AWS deployment plan documented (S3 for storage, Elastic Beanstalk for compute, EventBridge for scheduled rebuilds)

---

## Team

| Name | Role |
|------|------|
| Johnson Chuang | Semantic Search & Integration |
| Hooman Esteki | Data Processing & BM25 Implementation |

---

## References and Resources

- Dataset: https://amazon-reviews-2023.github.io/
- BM25 Algorithm: https://en.wikipedia.org/wiki/Okapi_BM25
- Sentence-Transformers: https://www.sbert.net/
- FAISS: https://faiss.ai/
- Streamlit: https://docs.streamlit.io/
- Groq: https://console.groq.com
- Groq Documentation: https://console.groq.com/docs

---

## License

MIT License
