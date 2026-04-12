# Dual Retrieval System: BM25 & Semantic Search
---

## Overview

This project implements and evaluates two complementary information retrieval methods on the Amazon Reviews 2023 dataset:

- BM25: Fast keyword-based ranking using Okapi BM25 algorithm
- Semantic Search: Meaning-based retrieval using sentence embeddings and FAISS indexing

The system demonstrates how statistical and neural approaches complement each other for effective document retrieval.

### About the Dataset

The project uses the **Amazon Reviews 2023** dataset (Books category), sourced from the McAuley Lab at UC San Diego. Two files are used:

- **`Books.jsonl.gz`** — User-written reviews, including star rating, review title, review text, and timestamp. Each record corresponds to one review of one product.
- **`meta_Books.jsonl.gz`** — Product metadata, including book title, author, description, features, price, and category. Each record corresponds to one product (identified by `parent_asin`).

The two files are joined on `parent_asin` to combine review text with book titles, forming the retrieval corpus.

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

- Model: `all-MiniLM-L6-v2` via sentence-transformers
- Index: FAISS `IndexFlatL2` (exact nearest-neighbour search over dense vectors)
- Embeddings: 384-dimensional vectors, one per document
- Complexity: O(n) brute-force L2 search; scales to larger corpora with approximate FAISS indexes
- Strengths: Captures semantic meaning, robust to synonyms and paraphrasing
- Limitations: Slower to build than BM25, less interpretable, sensitive to corpus coverage

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

#### Data Processing Details

The raw Books dataset is too large to load fully into memory, so it is read in chunks and filtered down to a stratified 20,000-review sample containing only records with meaningful text (minimum 20 characters). Each document in the corpus is formed by concatenating the product title (from metadata, joined on `parent_asin`) with the review text, giving retrieval models both keyword-rich title signals and richer semantic context from the review body. The processed corpus is saved as a Parquet file for reuse, alongside a pickle of the tokenized documents and an ASIN-to-title lookup table used to display human-readable results.

Two indexes are then built over the same 20K corpus. The BM25 index (built with `rank_bm25`) applies lowercasing, punctuation removal, and whitespace normalization before computing term-frequency scores, and is persisted as a pickle file for fast reload. The semantic index encodes each document into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer, then stores the embeddings in a FAISS `IndexFlatL2` structure for exact nearest-neighbour search. Both indexes are loaded at app startup, allowing queries to be served by either method without rebuilding from scratch.

---

## Key Findings

- **Keyword search works best for simple, specific queries.** When a user searches for something like "cookbook recipes" or "science fiction space", BM25 quickly finds books whose titles and reviews contain those exact words. It is fast and reliable for straightforward lookups.

- **Semantic search handles meaning better.** For vaguer queries like "guide for first time parents" or "historical fiction set in World War 2 from a female perspective", semantic search understood the intent and returned genuinely relevant books — even when the exact words weren't in the document. BM25 struggled here, often matching on individual words out of context (e.g. returning travel guides for the word "guide").

- **Both methods have blind spots.** Neither approach can return a book that isn't in the dataset. When we searched for "python programming", neither method found Python-specific books — because there simply weren't enough in our sample. No retrieval system can compensate for missing data.

- **Longer, complex queries favour semantic search.** BM25 treats a long query as a bag of individual words, which leads to false matches (e.g. "machine learning" matching sewing machine books). Semantic search encodes the full meaning of the query as a whole, making it more accurate for nuanced or multi-part requests.

- **A hybrid approach would likely perform best.** Each method covers the other's weaknesses. Combining them — or adding a reranking step — is the natural next direction for improving retrieval quality.

---

## Team

| Name | GitHub | Role |
|------|--------|------|
| Johnson Chuang | jchuang | Semantic Search & Integration |
| Hooman Esteki | esteki | BM25 Search & Integration |

---

## License

MIT License - See LICENSE file for details
