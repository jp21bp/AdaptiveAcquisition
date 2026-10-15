"""
Null Retriever
"""
from ..data.schemas import Document, Evidence, RetrievalResult

class NoRetriever:

    name = "none"

    def __init__(self):
        pass

    def retrieve(
        self,
        query: str,
        documents: list[Document],
        k: int
    ) -> RetrievalResult:
        return RetrievalResult(
            method=self.name,
            evidence=[],
            latency_ms=0,
            documents_considered=len(documents),
            documents_returned=0,
            retrieval_calls=0
        )