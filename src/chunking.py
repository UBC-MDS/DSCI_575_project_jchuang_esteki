
from typing import List

class DocumentChunker:
    """Split documents into chunks for context management."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, text: str) -> List[str]:
        """Split document into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks
