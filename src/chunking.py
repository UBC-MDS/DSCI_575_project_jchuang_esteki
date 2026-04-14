from typing import List


class DocumentChunker:
    """Chunk documents into manageable pieces."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize chunker with size and overlap parameters."""
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, text: str) -> List[str]:
        """Split document into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.overlap

        return chunks
