"""
Main pilot_v2 runner
"""
##### Retrievers
from typing import Literal
from ..src.retrieval.corpus import question_documents
from ..src.retrieval.all_retrievals import build_retriever
##### Custom data structures
from ..src.data.schemas import (
    Question, Evidence, AgentState
)
from ..src.evaluation.evaluator import Evaluator
##### Functions
from ..src.models.prompts import build_answer_prompt


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
            # Return unknown
            #'llm_answer' might need further preprocessing
                    # depending on the model's output structure
                # Note: file 'pilot_v2\src\evaluation\evaluator.py'
                        # expects 'llm_answer' to have a '.answer' field
        llm_answer = self.llm.generate(prompt)
        ## Evaluate LLM output
            # Associated: 'pilot_v2\src\evaluation\evaluator.py'
            # Returns EvaluationResult
        evaluation = self.evaluator.evaluate(
            llm_answer,
            question,
            evidences_retrieved
        )

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