"""
Creates the dense retriever 
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from ..data.schemas import Document, Evidence

class DenseRetriever:

    name='dense'

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = 'cpu'
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name, device=device
        )

    def retrieve(
        self,
        query: str,
        documents: list[Document],
        k: int
    ) -> list[Evidence]:
        # Gathering 'Context' concatenated sentences
        document_texts = [doc.text for doc in documents]
        # Embeddings
        if hasattr(self.model, 'encode_query'):
            # Query embedding
            query_embedding = \
                self.model.encode_query(
                    query, normalize_embeddings=True
                )
            # Documents embeddings
            document_embeddings = \
                self.model.encode_document(
                    document_texts, normalize_embeddings=True
                )
        else:
            query_embedding = \
                self.model.encode(
                    query, normalize_embeddings=True
                )
            document_embeddings = \
                self.model.encode(
                    document_texts, normalize_embeddings=True
                )
        # Query scored through matrix multiplication
            # Normalized embeds => dot product is cosimne similarity
        doc_embeds_arr = np.asarray(document_embeddings)
        query_embed_arr = np.asarray(query_embedding)
        scores = doc_embeds_arr @ query_embed_arr
        # Gathering the top k documents
        ranked_indices = np.argsort(scores)[::-1]
        selected = ranked_indices[:k]
        # Turning retrieved docs into 'Evidence' 
        return [
            Evidence(
                evidence_id=documents[i].document_id,
                document_id=documents[i].document_id,
                title=documents[i].title,
                text=documents[i].text,
                retrieval_method=self.name,
                rank=rank,
                retrieval_score=float(
                    scores[i]
                ),
            )
            for rank, i in enumerate(
                selected,
                start=1,
            )
        ]            
            
        







