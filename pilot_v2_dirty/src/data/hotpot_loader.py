"""
This file will load the HotpotQA dataset
"""
##### Libraries
#### General usage
import hashlib, json, os
from datasets import load_from_disk
#### Data structures
from .schemas import Context, Question, SupportingFact


##### Setup
#### Dataset path
DS_PATH = os.path.join(
    os.getcwd(),
    'downloads',
    'hotpotqa',
)
#### Assigning each Q a hash id
def stable_id(text: str) -> str:
    return hashlib.sha256(
        text.encode('utf-8')
    ).hexdigest()[:16]


##### Loader
def load_hotpotqa() -> list[Question]:

    hotpot_ds = load_from_disk(DS_PATH)
    hotpot_val = hotpot_ds['validation']

    questions = []

    for question_info in hotpot_val:
        contexts = []
        supporting_facts = []

        for title, sentences in zip(
            question_info['context']['title'],
            question_info['context']['sentences']
        ):
            contexts.append(
                Context(
                    title=title,
                    sentences=sentences
                )
            )

        for title, sentence_id in zip(
            question_info['supporting_facts']['title'],
            question_info['supporting_facts']['sent_id']
        ):
            supporting_facts.append(
                SupportingFact(
                    title=title,
                    sentence_id=sentence_id
                )
            )

        question_id = stable_id(
            question_info['question']
        )

        questions.append(
            Question(
                question_id=question_id,
                question=question_info['question'],
                answer=question_info['answer'],
                question_type=question_info['type'],
                difficulty=question_info['level'],
                contexts=contexts,
                supporting_facts=supporting_facts
            )
        )

    return questions


