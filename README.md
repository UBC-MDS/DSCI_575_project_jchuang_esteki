# Amazon Books Retrieval System
## Dual-Method Information Retrieval Foundation

A retrieval system combining **BM25 keyword-based search** and **semantic embedding-based search** to find relevant books from the Amazon Reviews 2023 dataset. This forms the foundation for a multi-phase Retrieval-Augmented Generation (RAG) project that will later add LLM-powered responses.

---

## Overview

### Project Goal

Building a retrieval system teaches us what keyword matching and semantic understanding can achieve independently. Users search in different ways: some use exact keywords ("mystery novel"), others describe intent ("book to help with anxiety"), and some have complex needs ("best book to learn machine learning with no math background"). This project explores how BM25 and semantic search handle these different patterns, then later milestones will combine them with LLM intelligence to generate helpful responses.

### Dataset: Amazon Reviews 2023

We use the **Amazon Reviews 2023** dataset from UC San Diego's McAuley Lab, containing 571 million+ reviews across 34 product categories collected from May 1996 through September 2023.

**Why "Books" Category:** Book reviews are detailed and substantive (users write paragraphs explaining what they liked), metadata is well-structured (clear titles, authors, descriptions), the dataset is large but manageable (11.7M reviews, 3.1M books), search queries are naturally diverse (by genre, topic, author, style, learning intent), and results are easy to verify (humans know what books are about).

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

---

## Getting Started

### Requirements
- Python 3.9+
- 8GB RAM
- 10GB disk space

### Setup and Installation

```bash
# Clone repository
git clone https://github.com/UBC-MDS/DSCI_575_project_jchuang_esteki.git
cd DSCI_575_project_jchuang_esteki

# Create and activate environment
conda env create -f environment.yml
conda activate 575-project
```

### Running the System

**First time (generate indexes):**
```bash
jupyter notebook

# Run notebooks in order:
# 01_exploration
# 02_data_preparation
# 03_bm25_keyword_search
# 04_semantic_embedding_search
# 05_evaluation_and_verification

# Note: This process can take up to 30 minutes (excluding raw data download).
# Note: Downloading the full dataset can take up to an hour, depending on your internet connection.
```

**After indexes are generated:**
```bash
streamlit run app/app.py
# Opens at http://localhost:8501
```

### Testing the System

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

#### Complex Queries (Both struggle):
```
  best book to learn machine learning with no math background
  historical fiction set in world war 2 from a female perspective
  self help book for overcoming procrastination and building better habits
```

**How to test:** Paste a query, try BM25 (fast, exact matches), then Semantic 
(slower, understands intent), then Hybrid (adjust weight slider). Compare 
results and see which method works better for different query types.

For detailed evaluation results, see `results/milestone1_discussion.md`.


---

## How the System Works

### Data Processing Pipeline

Raw data transforms through 5 distinct notebooks, each producing outputs for the next stage:

- **Notebook 01: Exploration** explores the dataset structure by loading sample records, inspecting fields and distributions, visualizing rating patterns, and documenting field selection rationale. This ensures all downstream decisions are grounded in data understanding.

- **Notebook 02: Data Preparation** processes millions of reviews into a retrieval-ready corpus. It uses DuckDB to efficiently stream 11.7 million reviews without memory overflow, creates a stratified 20,000-review sample balanced across ratings, joins reviews with metadata on parent_asin, applies consistent text preprocessing (lowercase, remove punctuation, normalize whitespace, filter < 20 chars), and concatenates product title with review text into unified documents. Outputs: `corpus.pkl` (20K preprocessed documents) and `books_sample.parquet`.

- **Notebook 03: BM25 Indexing** builds keyword-based search capability by loading the corpus, tokenizing documents consistently, building an inverted index using `rank_bm25`, and testing on sample queries. Output: `bm25_index.pkl`.

- **Notebook 04: Semantic Indexing** builds meaning-based search capability by loading the corpus, encoding all documents to 384-dimensional vectors using `all-MiniLM-L6-v2` sentence-transformers model, building a FAISS index for fast similarity search, and testing on sample queries. Output: `semantic_index/` directory.

- **Notebook 05: Evaluation** compares both methods on a diverse set of queries, creates retrieval results, analyzes performance patterns, and documents findings. Output: `results/milestone1_discussion.md`.

### Retrieval Methods

- **BM25 Keyword Search (Notebook 03):** Indexes every word and which documents contain it. When searching "mystery novel", finds documents with both words and scores based on term frequency and inverse document frequency. Speed: <10ms per query. Strengths: fast, transparent, exact keyword matching. Limitations: no synonyms ("anxious" ≠ "anxiety"), words treated independently, no intent understanding.

- **Semantic Search (Notebook 04):** Converts documents and queries into vectors where similar meaning corresponds to nearby locations. The pre-trained model learned that "parent", "parenting", "newborn", "baby" should be close in vector space. When searching "guide for first time parents", finds the query embedding's nearest neighbors and returns results ranked by distance. Speed: ~100ms per query. Strengths: understands intent, handles synonyms, context-aware. Limitations: slower, less transparent, may miss exact matches.

- **Hybrid Search (Optional):** Combines both methods with adjustable weights, giving users control to emphasize either keyword matching or semantic understanding.

---

## Project Structure

```
DSCI_575_project_jchuang_esteki/
│
├── README.md                         # Project documentation
├── environment.yml                   # Conda dependencies
├── .gitignore                        # Excludes data and credentials
│
├── data/
│   ├── raw/                          # Raw dataset (NOT in git)
│   │   ├── Books.jsonl.gz            # Reviews: 11.7M records
│   │   └── meta_Books.jsonl.gz       # Metadata: 3.1M records
│   └── processed/                    # Generated by notebooks (NOT in git)
│       ├── books_sample.parquet      # 20K stratified sample from 02
│       ├── corpus.pkl                # Preprocessed documents from 02
│       ├── bm25_index.pkl            # BM25 index from 03
│       └── semantic_index/           # FAISS index from 04
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   │   └─ Load and inspect dataset; justify field selection
│   │
│   ├── 02_data_preparation.ipynb
│   │   └─ Load with DuckDB → Stratified sample → Join → Preprocess → Combine
│   │
│   ├── 03_bm25_keyword_search.ipynb
│   │   └─ Tokenize → Build inverted index → Test queries
│   │
│   ├── 04_semantic_embedding_search.ipynb
│   │   └─ Encode documents → Build FAISS index → Test queries
│   │
│   └── 05_evaluation_and_verification.ipynb
│       └─ Load indexes → Retrieve top-5 for 10 queries → Analyze results
│
├── src/
│   ├── bm25.py                       # BM25Retriever class
│   ├── semantic.py                   # SemanticRetriever class
│   ├── utils.py                      # preprocess_text(), tokenize(), load functions
│   ├── retrieval_metrics.py          # Optional: evaluation metrics
│   └── hybrid.py                     # Optional: hybrid search
│
├── results/
│   └── milestone1_discussion.md      # Evaluation: 10 queries, comparisons, findings
│
├── app/
│   └── app.py                        # Streamlit web interface
│
└── .env                              # Environment variables (NEVER commit)
```
---
## File Connections

```
Raw Data (Books.jsonl.gz + meta_Books.jsonl.gz)
         ↓
    Notebook 01 (Explore)
         ↓
    Notebook 02 (Process: Load → Sample → Join → Preprocess → Combine)
         ↓
    corpus.pkl (20K documents)
         ↓
    ┌────────────────────┬────────────────────┐
    ↓                    ↓                    ↓
Notebook 03          Notebook 04         src/utils.py
(BM25 Index)     (Semantic Index)    (Preprocessing)
    ↓                    ↓
bm25_index.pkl    semantic_index/
    ├────────────────────┬────────────────────┤
    ↓                    ↓                    ↓
         Notebook 05 (Evaluate)
              ↓
    milestone1_discussion.md
              ↓
          app/app.py
     (Load indexes → Serve)
              ↓
        User Results
```

---

## Text Preprocessing

All text processing uses consistent functions from `src/utils.py`:

1. **Lowercase:** "The Da Vinci Code" → "the da vinci code" (case-insensitive matching)
2. **Remove Punctuation:** "don't" → "dont", "book?" → "book"
3. **Normalize Whitespace:** Multiple spaces → single space
4. **Filter Noise:** Remove documents < 20 characters

**Critical:** Documents and queries must be processed identically or retrieval fails. The same preprocessing applies when building indexes (Notebooks 03-04) and processing queries (Notebook 05 and app.py at runtime).

---

## Evaluation Results

Notebook 05 evaluates both retrieval methods on a diverse set of 10 queries spanning three difficulty levels: Easy queries like "mystery novel" test pure keyword matching; Medium queries like "book to help with anxiety" require semantic understanding; Complex queries like "best book to learn machine learning with no math background" challenge both methods with multiple constraints.

For each query, both methods retrieve top-5 results. The evaluation compares which method performs better on different query types and identifies cases where each method excels or fails.

**Key Findings:** BM25 excels on simple, exact keyword searches but gets confused by words with multiple meanings (e.g., "guide" returns travel guides when searching "guide for first time parents"). Semantic search better understands user intent and handles multi-word concepts (correctly interprets "machine learning" as one concept rather than confusing it with sewing machines). Both methods struggle when requested books don't exist in the dataset, when queries are vague or abstract, or when queries contain many specific requirements.

**Full Evaluation:** Complete results including all 10 queries with top-5 results from each method, detailed analysis of 5 queries, comparative findings, and recommendations are documented in `results/milestone1_discussion.md`.

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

---

## License

MIT License
