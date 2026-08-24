"""Shared data classes used across the pipeline."""


class Document:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata


class SearchResult:
    def __init__(self, document: Document, score: float):
        self.document = document
        self.score = score

    def __repr__(self):
        preview = self.document.content[:80]
        return f"SearchResult(score={self.score:.4f}, content='{preview}...')"


class Chunk:
    def __init__(self, page_start: int, page_end: int, title: str | None):  # page numbers are 1-based.
        self.page_start = page_start
        self.page_end = page_end
        self.title = title
        self.page_range = [n for n in range(page_start, page_end + 1)]
