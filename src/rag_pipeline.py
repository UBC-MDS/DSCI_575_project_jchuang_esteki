from typing import List, Dict, Tuple
from src.chunking import DocumentChunker
from src.prompts import RAGPrompts


class RAGPipeline:
    """RAG pipeline with hybrid retrieval and document tracking."""

    def __init__(self, bm25_retriever, semantic_retriever, llm, prompt_version="balanced"):
        """Initialize RAG pipeline with retrieval components and LLM."""
        self.bm25 = bm25_retriever
        self.semantic = semantic_retriever
        self.llm = llm
        self.corpus = None
        self.chunker = DocumentChunker(chunk_size=500, overlap=50)
        self.prompt_template = RAGPrompts.get_template(prompt_version)

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> Tuple[List[int], List[str]]:
        """Hybrid retrieval: combine BM25 and semantic search."""
        bm25_results = self.bm25.search(query, top_k)
        semantic_results = self.semantic.search(query, top_k)

        bm25_ids = set(idx for idx, _ in bm25_results)
        semantic_ids = set(idx for idx, _ in semantic_results)
        combined_ids = list(bm25_ids | semantic_ids)[:top_k]

        documents = [self.corpus[idx] for idx in combined_ids]
        return combined_ids, documents

    def build_context(self, documents: List[str], max_tokens: int = 2000) -> str:
        """Build context from documents with token limit."""
        context = ""
        token_count = 0

        for i, doc in enumerate(documents, 1):
            review_text = "Review " + str(i) + ": " + doc
            token_count += len(review_text.split())

            if token_count > max_tokens:
                break

            context += review_text + " "

        return context

    def generate(self, query: str, context: str) -> str:
        """Generate answer using LLM."""
        prompt = self.prompt_template.format(context=context, question=query)

        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return "Error: " + str(e)

    def invoke(self, query: str, top_k: int = 5) -> Dict:
        """Complete RAG pipeline: retrieve -> context -> generate."""
        doc_ids, documents = self.retrieve_hybrid(query, top_k)
        context = self.build_context(documents)
        answer = self.generate(query, context)

        return {
            "query": query,
            "documents_retrieved": len(documents),
            "context_length": len(context),
            "answer": answer,
            "retrieved_doc_ids": doc_ids
        }
