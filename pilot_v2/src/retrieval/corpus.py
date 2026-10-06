"""
Creates the corpus information available for retriever.

For pilot_v2, the only corpus available to the retriever will
        be the available 'Context' components within EACH question
    Ensures that each question will have the same candidate corpus
        This allows us to test effects of differente retrieval
                algorithms on the same question, same corpus

Note: EACH Q has 10 'Context' components
    Thus, each Q will have 10 available documents to retrieve from
"""

from ..data.schemas import Document, Question

def question_documents(question: Question) -> list[Document]:
    
    documents = []

    for i, context in enumerate(question.contexts):
        # Transforming 'Context' -> 'Document'
        documents.append(
            Document(
                document_id=f"{question.question_id}:{i}",
                title=context.title,
                sentences=context.sentences
            )
        )