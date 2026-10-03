"""
File contains the core data structures
"""
##### Importing libraries
from dataclasses import dataclass, field
from typing import Any

##### Evidence-based
@dataclass
class SupportingFact:
    title: str
    sentence_id: int

@dataclass
class Context:
    title: str
    sentences: list[str]

    @property
    def document_id(self) -> str:
        return self.title

@dataclass
class Evidence:
    evidence_id: str
    document_id: str
    title: str
    sentence_id: int
    text: str

    retrieval_method: str
    retrieval_score: float | None = None


##### QA-based
@dataclass
class Question:
    question_id: str
    question: str
    answer: str

    question_type: str
    difficulty: str

    contexts: list[Context] = field(
        default_factory=list
    )
    supporting_facts = list[SupportingFact] = field(
        default_factory=list
    )

@dataclass
class AnswerResult:
    answer: str
    self_reported_confidence: float | None

    input_tokens: int | None
    output_tokens: int | None

    latency_ms: float

    raw_output: str

@dataclass
class EvaluationResult:
    exact_match: float
    f1: float
    correct: bool

    supporting_facts_retrieved: int
    supporting_facts_total: int
    evidence_coverage: float


##### Environment-based
@dataclass
class AgentState:
    state_id: str
    question_id: str

    evidence_ids: list[str]

    answer: str | None
    self_reported_confidence: float | None

    correct: bool | None
    f1: float | None

    input_tokens: int
    output_tokens: int

    llm_calls: int
    retrieval_calls: int

    latency_ms: float

    evidence_coverage: float

@dataclass
class Transition: 
    transition_id: str

    question_id: str

    state_id: str
    next_state_id: str

    action_type: str

    utility_before: float
    utility_after: float
    delta_utility: float

    acquisition_cost: float

    cost_adjusted_value: float

    features_before: dict[str, Any]








