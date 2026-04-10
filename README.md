# Dual Retrieval System: BM25 & Semantic Search
---

## Overview

This project implements and evaluates two complementary information retrieval methods on the Amazon Reviews 2023 dataset:

- BM25: Fast keyword-based ranking using Okapi BM25 algorithm
- Semantic Search: Meaning-based retrieval using sentence embeddings and FAISS indexing

The system demonstrates how statistical and neural approaches complement each other for effective document retrieval.

---

## Quick Start

### 1. Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate 575-project
```

### 2. Data Preparation
Download the Amazon Reviews 2023 dataset from: https://amazon-reviews-2023.github.io/

Categories included:
- Books (primary ~5GB compressed)

Place raw data in:
```
data/raw/
├── Books.jsonl.gz
├── meta_Books.jsonl.gz
└── (other categories if desired)
```

### 3. Run Notebooks in Order
```bash
# Terminal 1: Launch Jupyter
jupyter notebook

# In browser, run notebooks in this order:
# 1. notebooks/01_exploration.ipynb
# 2. notebooks/02_data_preparation.ipynb
# 3. notebooks/03_bm25_keyword_search.ipynb
# 4. notebooks/04_semantic_embedding_search.ipynb
# 5. notebooks/05_evaluation_and_verification.ipynb
```

### 4. Launch Web App
```bash
streamlit run app/app.py
```

---

## Project Structure

```
DSCI_575_project_jchuang_esteki/
├── README.md                          # Project documentation
├── environment.yml                    # Conda environment specification
├── .env.example                       # Example environment variables
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── raw/                           # Raw dataset (gitignored)
│   │   ├── Books.jsonl.gz
│   │   └── meta_Books.jsonl.gz
│   └── processed/                     # Processed artifacts
│       ├── books_sample.parquet       # Filtered & deduplicated corpus
│       ├── corpus.pkl                 # Tokenized documents
│       ├── bm25_index.pkl             # BM25 inverted index
│       └── semantic_index/            # FAISS embeddings & metadata
│
├── notebooks/
│   ├── 01_exploration.ipynb                 # Data exploration & analysis
│   ├── 02_data_preparation.ipynb            # Build corpus & utilities
│   ├── 03_bm25_keyword_search.ipynb         # BM25 implementation & testing
│   ├── 04_semantic_embedding_search.ipynb   # Semantic search with embeddings
│   └── 05_evaluation_and_verification.ipynb # Compare BM25 vs semantic
│
├── src/
│   ├── __init__.py
│   ├── bm25.py                        # BM25Retriever class
│   ├── semantic.py                    # SemanticRetriever class
│   ├── retrieval_metrics.py           # Evaluation metrics
│   └── utils.py                       # Tokenization & preprocessing
│
├── results/
│   └── milestone1_discussion.md       # Findings & analysis
│
└── app/
    └── app.py                         # Streamlit web interface
```

---

## Technical Implementation

### BM25 Retriever
- Algorithm: Okapi BM25 (rank-bm25 library)
- Index: Inverted index with term frequency scores
- Complexity: O(n) per query, instant ranking
- Strengths: Fast, interpretable, keyword-precise
- Limitations: Cannot capture semantic similarity

### Semantic Retriever

coming soon

### Data Pipeline
```
Raw Dataset (5GB)
    |
[Chunked Loading - Memory Optimized]
    |
Text Filtering (minimum 20 chars, non-null)
    |
Corpus Creation (20K reviews)
    |
Tokenization (utils.py)
    |
Dual Indexing
├── BM25 Index (pkl)
└── FAISS Index + Embeddings (bin)
```

---

## Key Findings

coming soon ...

---

## Team

| Name | GitHub | Role |
|------|--------|------|
| Johnson Chuang | jchuang | Semantic Search & Integration |
| Hooman Esteki | esteki | BM25 Search & Integration |

---

## License

MIT License - See LICENSE file for details
