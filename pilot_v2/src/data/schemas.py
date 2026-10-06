"""
File contains the core data structures
"""
##### Importing libraries
from dataclasses import dataclass, field
from typing import Any

##### Evidence-based
@dataclass
class SupportingFact:
    # Used to store HotpotQA's supporting fact for each Q
        # Each supporting fact contains two fields: title and a sentence_id
    # Associated: '/src/data/hotpot_loader.py'
    title: str
    sentence_id: int

@dataclass
class Context:
    # Used to store HotpotQA's context for each Q
        # Each context contains two fields: 'title' and 'sentences'
    # Associated: '/src/data/hotpot_loader.py'
    title: str
    sentences: list[str]

    @property
    def document_id(self) -> str:
        return self.title

@dataclass
class Document:
    # Used to store the documents retrieved by retriever
        # In pilot_v2, these will be HotpotQA's contexts 
    document_id: str
        # Format: "{question's SHA encoding}: {retrieved rank??}"
    title: str
        # Context's title
    sentences: list[str]
        # Context's sentences

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

@dataclass
class Evidence:
    evidence_id: str
        # For pilot_v2: Will be the retrieved document's 'document_id'
        # Format: "{question's SHA encoding}: {retrieved rank??}"
    document_id: str
        # Format: "{question's SHA encoding}: {retrieved rank??}"
    title: str
        # Document's title == Context's title
    # sentence_id: int 
        # Retriever works with paragraphs, not sentences
    text: str
        # Document's (Context's) concatenated sentences

    retrieval_method: str
        # For pilot_v2: {BM25, dense, random}
    rank: int 
        # Order within the top k retrieved docs
    retrieval_score: float | None = None
        # Score assigned by each retriever


##### QA-based
@dataclass
class Question:
    # Associated: '/src/data/hotpot_loader.py'
    question_id: str
        # SHA encoding for the question itself
    question: str
        # HotpotQA question itself
    answer: str
        # HotpotQA answer

    question_type: str
        # HotpotQA type: {bridge, comparison, etc.}
    difficulty: str
        # HotpotQA difficulty: {easy, medium, hard}

    contexts: list[Context] = field(
        default_factory=list
    )
        #HotpotQA's context assigned to each question
    supporting_facts: list[SupportingFact] = field(
        default_factory=list
    )
        #HotpotQA's supporting fact assigned to each question

@dataclass
class AnswerResult:
    # Stores LLM's answer, which ideally has:
        # "ANSWER: .... \n CONFIDENCE: ..."
    # Associated: '/evaluation/parser.py'
    answer: str
        # LLM's 'ANSWER'
    self_reported_confidence: float | None
        # LLM's 'CONFIDENCE'

    input_tokens: int | None
        # Tokens in input query
    output_tokens: int | None
        # Tokens in output query

    latency_ms: float
        # Time it took to fulfill request

    raw_output: str
        # LLM's raw answer

@dataclass
class EvaluationResult:
    # Associated: 
        # '/src/evaluation/evaluator.py'
        # '/src/evaluation/eval_evidence.py'
    exact_match: float
        # LLM answer exactly matches HotpotQA answer
    f1: float
        # LLM's output F1 score with HotpotQA's answer
    correct: bool
        # Bool of 'exact_match'

    supporting_facts_retrieved: int
        # Intersection of retrieved docs and Hotpot's supporting info
    supporting_facts_total: int
        # Length of hotpot's supporting info
    evidence_coverage: float
        # Recall of hotpot's supporting info
            # Len(intersection)/len(supporting_facts)


##### Environment-based
@dataclass
class AgentState:
    # Associated: '/experiments/pilot_v2_exp.py'
    state_id: str
        # A unique uuid 
    question_id: str
        # SHA encoding of the question

    evidence_ids: list[str]
        # List of retrieved docs

    answer: str | None
        # LLM's answer
    self_reported_confidence: float | None
        # LLM's self-reported confidence

    correct: bool | None
        # LLM answer == HotpotQA answer
    f1: float | None
        # LLM's answer F1 score

    input_tokens: int
        # Input query tokens
    output_tokens: int
        # LLM's output tokens

    llm_calls: int
        # Num calls done to llm
    retrieval_calls: int
        # Num calls done to retriever

    latency_ms: float
        # LLM's response time

    evidence_coverage: float
        # Supporting fact recall:
            # Len(intersection)/len(supporting_facts)

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








