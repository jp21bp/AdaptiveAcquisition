"""
File loads in all frozen questions
"""
import json, yaml
from pathlib import Path
from ..data.schemas import Context, Question, SupportingFact

def load_questions() -> list[Question]:
    questions = []
    path = Path.cwd().joinpath(
        'data',
        'pilot_v2',
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