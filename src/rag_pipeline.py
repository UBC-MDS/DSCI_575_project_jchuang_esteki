from typing import List, Dict, Tuple
from src.chunking import DocumentChunker
from src.prompts import RAGPrompts
from src.hybrid import HybridRetriever


class RAGPipeline:
    """RAG pipeline using HybridRetriever for retrieval and an LLM for generation."""

    def __init__(self, bm25_retriever, semantic_retriever, llm, prompt_version="balanced"):
        """Initialize RAG pipeline with retrieval components and LLM."""
        self.hybrid = HybridRetriever(bm25_retriever, semantic_retriever)
        self.llm = llm
        self.corpus = None
        self.chunker = DocumentChunker(chunk_size=500, overlap=50)
        self.prompt_template = RAGPrompts.get_template(prompt_version)

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> Tuple[List[int], List[str]]:
        """Retrieve top-k documents using the hybrid retriever."""
        results = self.hybrid.search(query, top_k)
        doc_ids = [doc_id for doc_id, _ in results]
        documents = [self.corpus[doc_id] for doc_id in doc_ids]
        return doc_ids, documents

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
