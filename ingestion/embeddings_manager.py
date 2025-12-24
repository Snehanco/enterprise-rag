from typing import List, Dict, Any, Tuple
import logging
import numpy as np
from langchain.schema import Document
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """
    Unified embeddings manager handling:
    1. Dense (Semantic Search): BAAI/bge-small-en-v1.5
    2. Sparse (Keyword Search): Qdrant/bm25
    3. ColBERT (Reranking): colbert-ir/colbertv2.0
    """

    def __init__(
        self,
        dense_model: str = settings.DENSE_MODEL,
        colbert_model: str = settings.COLBERT_MODEL,
        sparse_model: str = "Qdrant/bm25",
    ):

        logger.info("Initializing EmbeddingsManager...")

        # 1. Dense Model
        self.dense_model = TextEmbedding(model_name=dense_model)

        # 2. Sparse Model
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model)

        # 3. ColBERT Model (Late Interaction)
        self.colbert_model = LateInteractionTextEmbedding(model_name=colbert_model)

        logger.info("All models initialized successfully.")

    def embed_dense(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generates dense vectors (e.g., 384 dimensions)."""
        # Convert generator to list of lists immediately
        embeddings = self.dense_model.embed(texts, batch_size=batch_size)
        return [emb.tolist() for emb in embeddings]

    def embed_sparse(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Dict[str, List[Any]]]:
        """Generates sparse vectors (indices and values)."""
        embeddings = self.sparse_model.embed(texts, batch_size=batch_size)
        return [
            {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
            for emb in embeddings
        ]

    def embed_colbert(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generates ColBERT multi-vector embeddings.
        Returns a list of numpy arrays. Each array has shape (Token_Count, 128).
        """
        # LateInteractionTextEmbedding returns a generator of numpy arrays
        return list(self.colbert_model.embed(texts, batch_size=batch_size))

    def embed_all(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Generates Dense, Sparse, and ColBERT embeddings for a list of texts."""
        dense = self.embed_dense(texts, batch_size)
        sparse = self.embed_sparse(texts, batch_size)
        colbert = self.embed_colbert(texts, batch_size)

        return [
            {"dense": d, "sparse": s, "colbert": c}
            for d, s, c in zip(dense, sparse, colbert)
        ]

    def embed_documents(
        self, documents: List[Document], batch_size: int = 32
    ) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Embeds LangChain Documents with all three formats.
        Useful for ingesting into a Vector DB or RAG pipeline.
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_all(texts, batch_size=batch_size)
        return list(
            zip(documents, embeddings)
        )  ## Return list of tuples (Document, embeddings(dict of all three))

    def embed_query(self, query_text: str) -> Dict[str, Any]:
        """
        Generates Dense, Sparse, and ColBERT embeddings for a single query string.
        Returns a dictionary with keys 'dense', 'sparse', and 'colbert'.
        """
        # Wrap single string in a list to satisfy the batch-processing methods
        query_list = [query_text]

        # 1. Dense: Get the first vector [384]
        dense_vec = self.embed_dense(query_list)[0]

        # 2. Sparse: Get the first dict {"indices": [], "values": []}
        sparse_vec = self.embed_sparse(query_list)[0]

        # 3. ColBERT: Get the first multi-vector [Token_Count, 128]
        # We convert the numpy array to a list for JSON compatibility with Qdrant
        colbert_vec = self.embed_colbert(query_list)[0].tolist()

        return {"dense": dense_vec, "sparse": sparse_vec, "colbert": colbert_vec}
