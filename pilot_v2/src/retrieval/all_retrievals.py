"""
Puts all the retrieval methods in one place
"""
from .random import RandomRetriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from typing import Union


def build_retriever(
    method:str, seed: int = 42
)-> Union[RandomRetriever, BM25Retriever, DenseRetriever]:

    if method=="random": return RandomRetriever(seed)
    if method=="bm25": return BM25Retriever()
    if method=="dense": return DenseRetriever(device='cpu')
    raise ValueError(f"Unknown retriever: {method}")




