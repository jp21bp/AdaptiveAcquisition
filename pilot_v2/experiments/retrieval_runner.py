"""
Runs a retrieval method
"""
import time
from ..src.data.schemas import Question
from ..src.retrieval.corpus import question_documents
from ..src.models.prompts import build_answer_prompt

from ..src.retrieval.random import RandomRetriever
from ..src.retrieval.bm25 import BM25Retriever
from ..src.retrieval.dense import DenseRetriever
from typing import Union

class RetrievalExperiment:

    def __init__(self, llm, evaluator):
        self.llm = llm
        self.evaluator = evaluator

    def run(
        self, 
        question: Question, 
        retriever: Union[
            RandomRetriever, BM25Retriever, DenseRetriever
        ],
        k:int
    ):
        # Turning Q's 'Context' into 'Document's
        documents = question_documents(question)
        # Gather 'list[Evidence]' from retrieval execution
        evidences_retrieved = retriever.retrieve(
            question.question,
            documents,
            k
        )
        # Invoking LLM
        prompt = build_answer_prompt(
            question.question,
            [e.text for e in evidences_retrieved]
        )
        answer = self.llm.generate(prompt)
        # Evaluate LLM output
            # Associated: '/src/evaluation/evaluation.py'
        evaluation = self.evaluator.evaluate(
            answer,
            question,
            evidences_retrieved
        )
        # Final answer
        return {
            "question_id": question.question_id,
            "retrieval_method": retriever.name,
            "k": k,
            "answer": answer.answer,
            "confidence": answer.self_reported_confidence,
            "f1": evaluation.f1,
            "correct": evaluation.correct,
            "evidence_coverage": evaluation.evidence_coverage,
        }














