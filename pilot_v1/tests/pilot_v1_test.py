"""
File tests the entire 'pilot_v1' pipeline with fake LLM
"""

from ..src.data.schemas import Context, Question
from ..src.models.fake_llm import FakeLLM
from ..experiments.pilot_v1_exp import ExperimentRunner

def test_pipeline():
    ### Fake question
    question = Question(
        question_id="test-1",
        question="Who wrote Hamlet?",
        answer="William Shakespeare",
        question_type="comparison",
        difficulty="easy",
        contexts=[
            Context(
                title="Hamlet",
                sentences=[
                    "Hamlet is a tragedy.",
                ],
            )
        ],
        supporting_facts=[],
    )
    ### Fake LLM for experiment
    runner = ExperimentRunner(
        llm=FakeLLM()
    )
    ### Results of fake experiment
    state = runner.run_state(question, evidence=[])
    ### Assertions to ensure structural integrity of experiment
    assert state.question_id == 'test-1'
    assert state.answer == 'FAKE LLM TEST'
    assert state.llm_calls == 1
    assert state.input_tokens > 0


if __name__ == "__main__":
    test_pipeline()
    print("Success")



