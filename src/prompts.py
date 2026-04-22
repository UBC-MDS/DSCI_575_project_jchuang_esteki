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

    @staticmethod
    def get_template(version: str = "balanced") -> str:
        """Get prompt template by version name."""
        templates = {
            "balanced": RAGPrompts.BALANCED,
            "strict": RAGPrompts.STRICT
        }
        return templates.get(version.lower(), RAGPrompts.BALANCED)
