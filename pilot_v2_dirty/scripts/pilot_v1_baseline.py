"""
File acts as the baseline, putting everything together
"""

import json, yaml
from pathlib import Path
from ..src.data.schemas import Context, Question, SupportingFact
from ..experiments.pilot_v1_exp import ExperimentRunner
from ..src.models.fake_llm import FakeLLM

def load_questions():
    questions = []
    path = Path.cwd().joinpath(
        'AdaptiveAcquisition',
        'data',
        'processed',
        'pilot_v1',
        'questions.jsonl'
    )
    for line in path.read_text().splitlines():
        raw = json.loads(line)

        contexts = [
            Context(**ctx)
            for ctx in raw['contexts']
        ]

        supporting_facts = [
            SupportingFact(**sf)
            for sf in raw['supporting_facts']
        ]

        questions.append(
            Question(
                question_id=raw['question_id'],
                question=raw['question'],
                answer=raw['answer'],
                question_type=raw['question_type'],
                difficulty=raw['difficulty'],
                contexts=contexts,
                supporting_facts=supporting_facts
            )
        )

    return questions

def main():
    questions = load_questions()

    runner = ExperimentRunner(llm=FakeLLM())

    for question in questions:
        state = runner.run_state(
            question, evidence=[]
        )

        print("=" * 60)
        print(question.question)
        print("ANSWER:", state.answer)
        print(
            "CONFIDENCE:",
            state.self_reported_confidence,
        )
        print("CORRECT:", state.correct)
        print("F1:", state.f1)
        print(
            "LATENCY:",
            state.latency_ms,
        )

if __name__ == "__main__":
    main()

