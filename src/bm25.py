import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from src.utils import tokenize

class BM25Retriever:
    """BM25 keyword-based retriever using Okapi BM25 algorithm."""

    def __init__(self):
        """Initialize empty BM25 retriever."""
        self.bm25 = None
        self.corpus = None

    def build_index(self, corpus: List[str]):
        """Build BM25 index from corpus."""
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus = corpus
        print(f"BM25 index built on {len(corpus)} documents")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for query in corpus. Returns (doc_id, score) pairs."""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def save(self, path: str):
        """Save BM25 index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.bm25, f)
        print(f"BM25 index saved to {path}")

    def load(self, path: str):
        """Load BM25 index from disk."""
        with open(path, 'rb') as f:
            self.bm25 = pickle.load(f)
        print(f"BM25 index loaded from {path}")
