"""
Random retriever, used as baseline
"""
import random, hashlib, time

from ..data.schemas import Document, Evidence, RetrievalResult

class RandomRetriever:

    name = "random"

    def __init__(self, seed: int = 42):
        self.seed=seed

    def retrieve(
        self,
        query: str,
        documents: list[Document],
        k: int
    ) -> list[Evidence]:
        # Starting latency timer
        start = time.perf_counter()
        # Creating a query SHA encoding
        query_sha = int(
            hashlib.sha256(
                query.encode('utf-8')
            ).hexdigest()[:8]
        )
        # Unique generator based on seed and query
            # Thus diff queries will retrieve diff docs
            # But same queries will retrieve same docs
        rand_generator = random.Random(
            self.seed + query_sha
        )
        # Corpus candidate
        candidates = documents.copy()
        rand_generator.shuffle(candidates)
        # Selecting top k retrieved docs
        selected = candidates[:k]
        # Turning retrieved docs into 'Evidence'
        evidence = [
            Evidence(
                evidence_id=doc.document_id,
                document_id=doc.document_id,
                title=doc.title,
                text=doc.text,
                retrieval_method=self.name,
                rank=rank,
                retrieval_score=None
            ) for rank, doc in enumerate(
                selected, start=1
            )
        ]    
        # Stopping timer
        latency_ms = (time.perf_counter() - start) * 1000
        # Turning retrieved documents into 'RetrievalResult'
        return RetrievalResult(
            method=self.name,
            evidence=evidence,
            latency_ms=latency_ms,
            documents_considered=len(documents),
            documents_returned=len(evidence),
            retrieval_calls=1
        )

