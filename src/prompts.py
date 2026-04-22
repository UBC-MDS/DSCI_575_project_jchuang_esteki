class RAGPrompts:
    """RAG prompt templates for different answer styles."""

    BALANCED = """Based on the following book reviews and information, answer the question.

Context:
{context}

Question: {question}

Answer:"""

    STRICT = """Using ONLY the provided book reviews and information below, answer the question.
If the information is insufficient to answer, say so.

Context:
{context}

Question: {question}

Answer:"""

    RECOMMENDATION = """You are a helpful book recommendation assistant. Based on the following book reviews, recommend books that match the user's request. For each book, briefly explain why it fits based on the reviews.

Context:
{context}

User request: {question}

Answer:"""

    @staticmethod
    def get_template(version: str = "balanced") -> str:
        """Get prompt template by version name."""
        templates = {
            "balanced": RAGPrompts.BALANCED,
            "strict": RAGPrompts.STRICT,
            "recommendation": RAGPrompts.RECOMMENDATION
        }
        return templates.get(version.lower(), RAGPrompts.BALANCED)
