"""
Evalutes the retrieval method
"""
from ..data.schemas import Evidence, Question

def document_recall(
    evidence: list[Evidence],
    question: Question
) -> float:
    # Retrieving information
    retrieved_titles = {e.title for e in evidence}
    true_titles = {sf.titles for sf in question.supporting_facts}
    # Returning recall
    if not true_titles: return 0.0
    intersect = retrieved_titles & true_titles
    return len(intersect)/len(true_titles)






