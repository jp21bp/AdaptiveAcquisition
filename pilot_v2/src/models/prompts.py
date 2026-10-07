"""
File contains the standard prompt for the model
"""

def build_answer_prompt(
    question: str,
    evidence: list[str]
):
    ### Formatting evidence
    evidence_text = "\n\n".join(
        f'[Document {i+1}] -- {text}'
        for i, text in enumerate(evidence)
    )

    return f"""
Answer the question using the provided documents.

Question:
{question}

Evidence:
{evidence_text if evidence_text else 'No external evidence was provided.'}

Instructions:
- Give a concise answer.
- Use the documents as evidence.
- Do not assume that every document is relevant.
- Do not invent unsupported information.
- If the documents do not provide enough information, state that explicitly.

Return exactly the following format:

ANSWER: <answer>
CONFIDENCE: <number from 0 to 100>
"""
