from typing import List, Tuple


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic embedding search.

    Uses round-robin interleaving to merge ranked results from both retrievers,
    returning documents with source attribution (BM25/Semantic/BM25 + Semantic).
    """

    def __init__(self, bm25_retriever, semantic_retriever):
        """Initialize with a BM25 retriever and a semantic retriever.

        Parameters
        ----------
        bm25_retriever : BM25Retriever
            An instance exposing a ``search(query, top_k)`` method that returns
            ``(doc_id, score)`` tuples.
        semantic_retriever : SemanticRetriever
            An instance exposing the same ``search(query, top_k)`` interface.
        """
        self.bm25 = bm25_retriever
        self.semantic = semantic_retriever

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, str]]:
        """Search using both retrievers and merge via round-robin interleaving.

        Alternates picking from each method's ranked list, deduplicating as it goes,
        until top-k results are collected. Returns a list of (doc_id, source) tuples
        where source is 'BM25', 'Semantic', or 'BM25 + Semantic'.
        """
        bm25_results = self.bm25.search(query, top_k)
        semantic_results = self.semantic.search(query, top_k)

        bm25_ids = [idx for idx, _ in bm25_results]
        semantic_ids = [idx for idx, _ in semantic_results]
        overlap = set(bm25_ids) & set(semantic_ids)

        # Round-robin interleave: alternate picking from BM25 and semantic so both
        # methods are fairly represented in the final top_k. Without this, Python's
        # dict insertion order would mean BM25 results always fill the top slots first,
        # leaving no room for semantic-only results.
        combined = []
        seen = set()
        for bm25_id, semantic_id in zip(bm25_ids, semantic_ids):
            for doc_id, id_list in ((bm25_id, bm25_ids), (semantic_id, semantic_ids)):
                if doc_id not in seen:
                    source = (
                        "BM25 + Semantic" if doc_id in overlap
                        else ("BM25" if id_list is bm25_ids else "Semantic")
                    )
                    combined.append((doc_id, source))
                    seen.add(doc_id)
                if len(combined) == top_k:
                    return combined

        return combined