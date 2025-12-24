import logging
from core.vectorstore import QdrantManager
from core.decorators import log_execution
from ingestion.chunking import DocumentChunker
from ingestion.document_loader import DocumentLoader
from ingestion.embeddings_manager import EmbeddingsManager
from ingestion.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class IngestionPipeline:
    @log_execution
    def process_document(
        self,
        file_path: str,
        document_name: str,
        collection_name: str,
    ):
        docs = []
        doc_loader = DocumentLoader()
        file_exists = doc_loader.validate_file_path(file_path)
        if not file_exists:
            raise FileNotFoundError(f"File not found or invalid: {file_path}")
        else:
            docs = doc_loader.load_document(file_path, document_name)
        text_cleaner = TextCleaner()
        cleaned_docs = text_cleaner.clean_documents(docs, language="en")
        document_chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
        chunks = document_chunker.chunk_documents(cleaned_docs)
        embedding_manager = EmbeddingsManager()
        final_chunks = embedding_manager.embed_documents(chunks)
        qdrant_manager = QdrantManager()
        qdrant_manager.upsert_points(collection_name, final_chunks)
        logger.info(f"Document processing completed for {file_path}")
