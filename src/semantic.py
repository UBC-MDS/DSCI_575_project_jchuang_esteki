import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """Semantic retriever using sentence embeddings and FAISS for nearest-neighbour search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence-transformers model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus = None

    def build_index(self, corpus: List[str]):
        """Encode corpus into embeddings and build a FAISS index."""
        print(f"Encoding {len(corpus)} documents...")
        embeddings = self.model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.corpus = corpus
        print(f"FAISS index built with {self.index.ntotal} vectors (dim={dim})")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for query. Returns (doc_id, distance) pairs (lower distance = more similar)."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        query_vector = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]

    def save(self, path: str):
        """Save FAISS index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)
        print(f"FAISS index saved to {path}")

    def load(self, path: str):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path)
        print(f"FAISS index loaded from {path}")
