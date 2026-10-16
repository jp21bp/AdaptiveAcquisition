"""
File will parse LLM answers

Chosen model: "distilbert/distilbert-base-cased-distilled-squad"

Function won't be used in this pilot_v2
    Model chosen only returns context chars that best answer Q
"""
import re

def parse_answer(raw_output: str) -> tuple[str, float | None]:
    ### Regexes to be found
    ## For answer
    answer_match = re.search(
        r"ANSWER:\s*(.*?)(?:\n|$)",
        raw_output,
        flags=re.IGNORECASE
    )
    ## For confidence
    confidence_match = re.search(
        r'CONFIDENCE:\s*([0-9]+(?:\.[0-9]+)?)',
        raw_output,
        flags=re.IGNORECASE
    )

    ### Applying regex
    ## Answer
    answer = (
        answer_match.group(1).strip()
        if answer_match
        else raw_output.strip()
    )
    ## Confidence
    confidence = (
        float(confidence_match.group(1))
        if confidence_match
        else None
    )

    return answer, confidence



