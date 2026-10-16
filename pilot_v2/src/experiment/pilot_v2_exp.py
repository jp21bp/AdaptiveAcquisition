"""
Pilot_v2 runner with FakeLLM
"""
##### Custom data structures
import uuid
from ..data.schemas import(
    Question, AgentState, AnswerResult, Evidence
)
##### Retriever modules
from typing import Literal
from ..retrieval.corpus import question_documents
from ..retrieval.all_retrievals import build_retriever
##### Evaluator
from ..evaluation.evaluator import Evaluator
##### LLM
from ..models.fake_llm import FakeLLM


class ExperimentRunner:
    
    def __init__(
            self, 
            llm, 
            retriver_type: Literal["none", "random", "bm25", "dense"],  
            k: int
    ):
        self.llm = (llm or FakeLLM())
        ## Prepare the retriever
            # Associated: 'pilot_v2\src\retrieval\all_retrievals.py'
            # Returns Union[RandomRetriever, BM25Retriever, DenseRetriever]
        self.retriever = build_retriever(retriver_type)
        self.k = k
        self.evaluator = Evaluator()

    def run_state(
        self,
        question: Question
    ) -> AgentState:
        ## Turning Q's 'Context' -> 'Document'
            # Associated: 'pilot_v2\src\retrieval\corpus.py'
            # Returns List[Document]
        documents = question_documents(question)
        ## Invoking the retriever
            # Associated: 'pilot_v2\src\retrieval\{bm25,dense,random}.py'
            # Returns RetrievalResult 
        retrieval_result = self.retriever.retrieve(
            question.question,
            documents,
            self.k
        )
        ## Extracting the list of evidence
            # Associated: 'pilot_v2\src\retrieval\{bm25,dense,random}.py'
            # Returns List[Evidence]
        evidences_retrieved = retrieval_result.evidence
        ## Invoking LLM
            # Associated: 'pilot_v2\src\models\fake_llm.py'
            # Associated: 'pilot_v2\src\models\llm_invocation.py'
            # Return AnswerResult
        context = "".join(e.text for e in evidences_retrieved)
        llm_result = self.llm.generate(question, context)
        ## Evaluate LLM output
            # Associated: 'pilot_v2\src\evaluation\evaluator.py'
            # Returns EvaluationResult
        evaluation = self.evaluator.evaluate(
            llm_result,
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
            answer=llm_result.answer,
            self_reported_confidence=llm_result.self_reported_confidence,
            correct=evaluation.correct,
            f1=evaluation.f1,
            input_tokens=llm_result.input_tokens or 0,
            output_tokens=llm_result.output_tokens or 0,
            llm_calls=1,
            retrieval_calls=retrieval_result.retrieval_calls,
            latency_ms=llm_result.latency_ms,
            evidence_coverage=evaluation.evidence_coverage
        )
        return {
            "experiment_id" :"pilot_v2",
            "question_id": question.question_id,
            "model_name": "distilbert/distilbert-base-cased-distilled-squad",
            "retrieval_method": self.retriever.name,
            "k": self.k,
            "answer": llm_result.answer,
            "gold_answer": question.answer,
            "exact_match": evaluation.exact_match,
            "f1": evaluation.f1,
            "correct": evaluation.correct,
            "document_recall": evaluation.evidence_recall,
            "evidence_coverage": evaluation.evidence_recall,
            "retrieved_documents": evidences_retrieved,
            "retrieval_latency_ms": retrieval_result.latency_ms,
            "input_tokens": llm_result.input_tokens,
            "context_tokens":llm_result.context_tokens,
            "output_tokens": llm_result.output_tokens,
            "llm_latency_ms": llm_result.latency_ms,
            "llm_calls": 1,
            "agent_state": agent_state
        }                   