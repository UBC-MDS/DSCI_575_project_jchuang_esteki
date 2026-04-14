
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class SemanticRetriever:
    """Semantic search using FAISS and sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus = None

    def build_index(self, corpus: List[str]) -> None:
        """Encode corpus and build FAISS index."""
        self.corpus = corpus
        print(f"Encoding {len(corpus)} documents...")
        embeddings = self.model.encode(corpus, show_progress_bar=True, batch_size=32)
        embeddings = embeddings.astype('float32')

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"Index built: {self.index.ntotal} vectors (dim={embeddings.shape[1]})")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for similar documents."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_embedding = self.model.encode([query], show_progress_bar=False).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]

    def save(self, path: str) -> None:
        """Save index to disk."""
        faiss.write_index(self.index, str(path))
        print(f"Index saved to {path}")

    def load(self, path: str) -> None:
        """Load index from disk."""
        self.index = faiss.read_index(str(path))
        print(f"Index loaded from {path}")
