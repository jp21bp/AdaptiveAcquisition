"""
File will be used to evaluate the coverage of the 
    retrieved evidence against what HotpotQA grounded 
    as the supporting evidence for a given question.

For now it only uses recall to measure coverage.

In future experiments, will evaluate if necessary to
    also use precision and F1 score
"""
from ..data.schemas import Evidence, Question

##### Evaluating coverage of retrieved evidence
def evidence_coverage(
    evidence: list[Evidence],
    question: Question
) -> tuple[int, int, float]:
    #### Grouping retrieved data's metadata
    retrieved = {
        (e.title, e.sentence_id)
        for e in evidence
    }
    #### Grouping supporting data's metada
    supporting = {
        (sf.title, sf.sentence_id)
        for sf in question.supporting_facts
    }
    #### Edge case: HotpotQA decided no coverage is necessary
    if not supporting: return 0, 0, 0.0
    #### Checking overlapping metadata sets
    found = retrieved & supporting
    #### Coverage = (overlapMetadata)/(supportingFacts)
        # Similar to 'recall' in F1 score
    return(
        len(found), 
        len(supporting),
        len(found)/len(supporting)
    )







