
from typing import List, Dict
from src.chunking import DocumentChunker
from src.prompts import RAGPrompts

class RAGPipeline:
    """RAG pipeline with chunking and hybrid retrieval."""

    def __init__(self, bm25_retriever, semantic_retriever, llm, prompt_version="balanced"):
        self.bm25 = bm25_retriever
        self.semantic = semantic_retriever
        self.llm = llm
        self.corpus = None
        self.chunker = DocumentChunker(chunk_size=500, overlap=50)
        self.prompt_template = RAGPrompts.get_template(prompt_version)

    def retrieve_hybrid(self, query, top_k=5):
        """Hybrid retrieval: combine BM25 and semantic."""
        bm25_results = self.bm25.search(query, top_k)
        semantic_results = self.semantic.search(query, top_k)
        bm25_ids = set(idx for idx, _ in bm25_results)
        semantic_ids = set(idx for idx, _ in semantic_results)
        combined_ids = list(bm25_ids | semantic_ids)[:top_k]
        return [self.corpus[idx] for idx in combined_ids]

    def build_context(self, documents, max_tokens=2000):
        """Build context from documents."""
        context = ""
        token_count = 0
        for i, doc in enumerate(documents, 1):
            review_text = "Review " + str(i) + ": " + doc
            token_count += len(review_text.split())
            if token_count > max_tokens:
                break
            context += review_text + " "
        return context

    def generate(self, query, context):
        """Generate answer using LLM."""
        prompt = self.prompt_template.format(context=context, question=query)
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return "Error: " + str(e)

    def invoke(self, query, top_k=5):
        """Complete RAG pipeline: retrieve and generate."""
        documents = self.retrieve_hybrid(query, top_k)
        context = self.build_context(documents)
        answer = self.generate(query, context)
        return {"query": query, "documents_retrieved": len(documents), "context_length": len(context), "answer": answer}
