"""
File will combine 'eval_answer' and 'eval_evidence' succintly.

Current implementation is basic, but future versions can 
    change the different methods in order to catpure more
    granular information from the experiment.
"""
from ..data.schemas import (
    AnswerResult, Evidence, EvaluationResult, Question
)
from .eval_answer import exact_match, token_f1
from .eval_retrieval import document_recall

##### Class to organize information
class Evaluator:

    def evaluate(
        self,
        prediction: AnswerResult,
        question: Question, 
        evidence: list[Evidence]
    ) -> EvaluationResult:
        ### Checking if prediction is an exact match
        exact = exact_match(
            prediction.answer,
            question.answer
        )
        ### Prediction's F1 score
        f1 = token_f1(
            prediction.answer,
            question.answer
        )
        ### Measuring coverage
            # In pilot_v2, evidence is retrieved at document granularity
                # Not at the sentence granularity
        # len_overlap, len_supporting, evidence_recall = \
        #     evidence_coverage(evidence, question)     # sentece - granularity
        evidence_recall = document_recall(evidence, question)

        ### Combining results
        return EvaluationResult(
            exact_match=exact,
            f1=f1,
            correct=bool(exact),
            # supporting_facts_retrieved=len_overlap,   #Sentence granularity
            # supporting_facts_total=len_supporting,    #Sentence granularity
            # evidence_coverage=evidence_recall         #Sentence granularity
            evidence_recall=evidence_recall
        )

    






