"""
File executes the 'pilot_v1' experiment
"""
import time
import uuid
from ..src.data.schemas import (
    AgentState, Evidence, Question
)
from ..src.evaluation.evaluator import Evaluator
from ..src.models.prompts import build_answer_prompt

##### Class to execute experiment
class ExperimentRunner:
    ### Initialize with model
    def __init__(self, llm):
        self.llm = llm
        self.evaluator = Evaluator()
    ### Run experiment
    def run_state(
        self,
        question: Question,
        evidence: list[Evidence]
    ) -> AgentState:
        ## Creating the prompt
        prompt = build_answer_prompt(
            question.question,
            [e.text for e in evidence]
        )
        ## Passing the prompt to the LLM
            # Returns an 'AnswerResult' object
        prediction = self.llm.generate(prompt)
        ## Evalute the results
        evaluation = self.evaluator.evaluate(
            prediction, question, evidence
        )
        ## Transferring results into AgentState
        return AgentState(
            state_id=str(uuid.uuid4()),
            question_id=question.question_id,
            evidence_ids=[e.evidence_id for e in evidence],
            answer=prediction.answer,
            self_reported_confidence=prediction.self_reported_confidence,
            correct=evaluation.correct,
            f1=evaluation.f1,
            input_tokens=prediction.input_tokens or 0,
            output_tokens=prediction.output_tokens or 0,
            llm_calls=1,
            retrieval_calls=0,
            latency_ms=prediction.latency_ms,
            evidence_coverage=evaluation.evidence_coverage
        )




