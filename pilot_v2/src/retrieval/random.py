"""
Random retriever, used as baseline
"""
import random, hashlib

from ..data.schemas import Document, Evidence

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
        return [
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
