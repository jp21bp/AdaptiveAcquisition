"""
File contains the standard prompt for the model
"""

def build_answer_prompt(
    question: str,
    evidence: list[str]
):
    ### Formatting evidence
    evidence_text = "\n\n".join(
        f'[Evidence {i+1}] -- {text}'
        for i, text in enumerate(evidence)
    )

    return f"""
You are answering a question using the supplied evidence.

Question:
{question}

Evidence:
{evidence_text if evidence_text else 'No external evidence was provided.'}

Instructions:
1. Answer the question as accurately as possible.
2. Use the evidence when it is available.
3. Do not invent facts that are unsupported by the evidence.
4. If the evidence is insufficient, say so.

Return exactly the following format:

ANSWER: <short answer>
CONFIDENCE: <number from 0 to 100>
"""
