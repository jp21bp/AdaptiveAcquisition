"""
Evalutes the retrieval method
Specfically, of the retrieved Documents, how many
        of them were part of the 'Question.supporting_facts'
        string of documents
"""
from ..data.schemas import Evidence, Question

def document_recall(
    evidence: list[Evidence],
    question: Question
) -> float:
    # Retrieving information
    retrieved_titles = {e.title for e in evidence}
    gold_titles = {sf.title for sf in question.supporting_facts}
    # Returning recall
    if not gold_titles: return 0.0
    intersect = retrieved_titles & gold_titles
    return len(intersect)/len(gold_titles)






