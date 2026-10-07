"""
Gathering the results associated with a retrieval method
"""
from dataclasses import dataclass
from ..data.schemas import Evidence

@dataclass
class RetrievalResult:

    method:str
    evidence: list[Evidence]
    latency_ms: float
    documents_considered: int
    documents_returned: int
    retrieval_calls: int




