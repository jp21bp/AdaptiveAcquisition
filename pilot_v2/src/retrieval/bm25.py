"""
Implements BM25 retriever
"""
import re, time
from rank_bm25 import BM25Okapi
from ..data.schemas import Document, Evidence, RetrievalResult

##### Tokenize text, needed for BM25 retriever
def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

##### BM25 class
class BM25Retriever:

    name = 'bm25'

    def retrieve(
        self,
        query: str,
        documents: list[Document],
        k: int
    ) -> RetrievalResult:
        # Starting latency timer
        start = time.perf_counter()
        # Tokenizing documents and query
        tokenized_docs = [tokenize(doc.text) for doc in documents]
            # Recall: 'doc.text' = concatenation of all sentences in 'Context.sentences'
        query_tokens = tokenize(query)
        # Intializing retriever
        bm25 = BM25Okapi(tokenized_docs)
        # Score all documents based on query
            # Return length = len(documents)
        scores = bm25.get_scores(query_tokens)
        # Ranking documents
        ranked_indices = sorted(
            range(len(documents)),
            key = lambda doc: scores[doc],
            reverse=True
        )
        # Selecting top k documents
        selected = ranked_indices[:k]
        # Transformando 'selcted' into 'list[Evidence]'
        evidence = [
            Evidence(
                evidence_id=documents[i].document_id,
                document_id=documents[i].document_id,
                title=documents[i].title,
                text=documents[i].text,
                retrieval_method=self.name,
                rank=rank,
                retrieval_score=float(scores[i])
            ) for rank, i in enumerate(selected, start=1)
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




