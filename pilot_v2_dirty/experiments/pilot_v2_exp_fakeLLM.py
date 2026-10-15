"""
Main pilot_v2 runner
"""
##### Retrievers
from typing import Literal
from ..src.retrieval.corpus import question_documents
from ..src.retrieval.all_retrievals import build_retriever
from ..src.evaluation.eval_retrieval import document_recall
##### Custom data structures
import uuid
from ..src.data.schemas import (
    Question, AgentState, AnswerResult
)
from ..src.evaluation.evaluator import Evaluator
##### Functions
from ..src.models.prompts import build_answer_prompt
##### Fake LLM
from ..src.models.fake_llm import FakeLLM

class ExperimentRunner:
    def __init__(self, llm, k: int):
        self.llm = llm
        self.evaluator = Evaluator()
        self.k = k

    def run_state(
        self,
        question: Question,
        retriever_type: Literal["none", "random", "bm25", "dense"]
    ) -> AgentState:
        ## Turning Q's 'Context' -> 'Document'
            # Associated: 'pilot_v2\src\retrieval\corpus.py'
            # Returns List[Document]
        documents = question_documents(question)
        ## Prepare the retriever
            # Associated: 'pilot_v2\src\retrieval\all_retrievals.py'
            # Returns Union[RandomRetriever, BM25Retriever, DenseRetriever]
        retriever = build_retriever(retriever_type)
        ## Invokiing the retriever
            # Associated: 'pilot_v2\src\retrieval\{bm25,dense,random}.py'
            # Returns RetrievalResult 
        retrieval_result = retriever.retrieve(
            question.question,
            documents,
            self.k
        )  
        ## Extracting the list of evidence
            # Associated: 'pilot_v2\src\retrieval\{bm25,dense,random}.py'
            # Returns List[Evidence]
        evidences_retrieved = retrieval_result.evidence
        ## Preparing LLM prompt
            # Assocaited: 'pilot_v2\src\models\prompts.py'
            # Returns str
        prompt = build_answer_prompt(
            question.question,
            [e.text for e in evidences_retrieved]
        )
        ## Invoking LLM
            # Associated: 'pilot_v2\src\models\fake_llm.py'
            # Return AnswerResult
            #'fake_llm.py' might need further preprocessing
                    # depending on the model's output structure
        # llm_answer = self.llm.generate(prompt)
        llm_invoked = FakeLLM().generate(prompt)
        ## Evaluate LLM output
            # Associated: 'pilot_v2\src\evaluation\evaluator.py'
            # Returns EvaluationResult
        evaluation = self.evaluator.evaluate(
            llm_invoked,
            question,
            evidences_retrieved
        )
        agent_state = AgentState(
            state_id=str(uuid.uuid4()),
            question_id=question.question_id,
            evidence_ids=[
                e.evidence_id for e
                in evidences_retrieved
            ],
            answer=llm_invoked.answer,
            self_reported_confidence=llm_invoked.self_reported_confidence,
            correct=evaluation.correct,
            f1=evaluation.f1,
            input_tokens=llm_invoked.input_tokens or 0,
            output_tokens=llm_invoked.output_tokens or 0,
            llm_calls=1,
            retrieval_calls=retrieval_result.retrieval_calls,
            latency_ms=llm_invoked.latency_ms,
            evidence_coverage=evaluation.evidence_coverage
        )
        return {
            "experiment_id" :"pilot_v2",
            "question_id": question.question_id,
            "model_name": "distilbert/distilbert-base-cased-distilled-squad",
            "retrieval_method": retriever.name,
            "k": self.k,
            "answer": llm_invoked.answer,
            "gold_answer": question.answer,
            "exact_match": evaluation.exact_match,
            "f1": evaluation.f1,
            "correct": evaluation.correct,
            "document_recall": document_recall(
                evidences_retrieved,
                question
            ),
            "evidence_coverage": evaluation.evidence_coverage,
            "retrieved_documents": evidences_retrieved,
            "retrieval_latency_ms": retrieval_result.latency_ms,
            "input_tokens": llm_invoked.input_tokens,
            "output_tokens": llm_invoked.output_tokens,
            "llm_latency_ms": llm_invoked.latency_ms,
            "llm_calls": 1,
            # "context_tokens": 
        }


{
  "experiment_id": "milestone2_v1",
  "question_id": "...",

  "model_name": "...",

  "retrieval_method": "bm25",
  "k": 2,

  "answer": "...",
  "gold_answer": "...",

  "exact_match": 1.0,
  "f1": 1.0,
  "correct": true,

  "document_recall": 1.0,
  "evidence_coverage": 1.0,

  "retrieved_documents": [
    "The Oberoi Group",
    "Oberoi family"
  ],

  "retrieval_latency_ms": 0.12,

  "input_tokens": 600,
  "output_tokens": 12,

  "llm_latency_ms": 1200,
  "llm_calls": 1,

  "context_tokens": 420
}