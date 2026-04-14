
class RAGPrompts:
    """RAG prompt templates."""

    BALANCED = """You are a helpful assistant answering questions about books based on customer reviews.

Based on the reviews below, answer the user's question. Be concise and helpful.

Context from reviews:
{context}

Question: {question}

Answer:"""

    STRICT = """Answer ONLY using information from the provided reviews.
Do not use any other knowledge. If the answer is not in the reviews, say so.

Reviews:
{context}

Question: {question}

Answer:"""

    @staticmethod
    def get_template(version="balanced"):
        templates = {"balanced": RAGPrompts.BALANCED, "strict": RAGPrompts.STRICT}
        return templates.get(version.lower(), RAGPrompts.BALANCED)
