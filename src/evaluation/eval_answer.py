"""
File normalizes LLM answers.

Currently using a simplified normalizer.

Eventually will use HotpotQA's official evaluation conventions.
"""

import re, string

##### Normalizer
def normalize_answer(text: str) -> str:
    #### Minimizing all text
    text = text.lower()

    #### Eliminating all punctuations
    text = re.sub(
        rf"[{re.escape(string.punctuation)}]",
        " ",
        text
    )

    #### Eliminating 'a','an', 'the'
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    #### Eliminating newlines and unnecessary whitespaces
    text = " ".join(text.split())

    return text


##### Checking if prediction and answer are exact matches
def exact_match(
    prediction: str,
    true: str
) -> float:
    return float(
        normalize_answer(prediction)
        == normalize_answer(true)
    )


##### Evaluating Tokens between pred and output
def token_f1(
    prediction: str,
    true: str
) -> float:
    #### Extracting tokens
    pred_tokens = normalize_answer(prediction).split()
    true_tokens = normalize_answer(true).split()

    #### Edge case - either one is empty
    if not pred_tokens or not true_tokens:
        return float(pred_tokens == true_tokens)

    #### Examining the common tokens
    common = set(pred_tokens) & set(true_tokens)
    if not common: return 0.0

    #### Calculating token f1 score
    precision = (len(common)/len(pred_tokens))
    recall = (len(common)/len(true_tokens))
    return (2 * precision * recall /(precision + recall))






