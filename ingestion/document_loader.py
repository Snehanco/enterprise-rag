from typing import List, Tuple
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import zipfile, tempfile
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:

    def __init__(self):
        self.SUPPORTED_FORMATS = {".pdf", ".zip"}

    def load_pdf(self, file_path: str, document_name: str = None) -> List[Document]:
        logger.info(f"Loading PDF: {file_path}")

        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()

            file_name = Path(file_path).name
            doc_name = document_name or file_name.replace(".pdf", "")

            for doc in documents:
                doc.metadata["file_name"] = file_name
                doc.metadata["document_name"] = doc_name
                doc.metadata["source_type"] = "pdf"
                if "page" in doc.metadata:
                    doc.metadata["page_number"] = doc.metadata["page"] + 1

            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {str(e)}")
            raise

    def load_zip(self, file_path: str, document_name: str = None) -> List[Document]:
        logger.info(f"Loading ZIP: {file_path}")

        documents = []
        zip_name = Path(file_path).stem  # Filename without extension
        doc_name = document_name or zip_name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                temp_path = Path(temp_dir)
                pdf_files = list(temp_path.rglob("*.pdf"))

                logger.info(f"Found {len(pdf_files)} PDFs in ZIP")

                for pdf_file in pdf_files:
                    try:
                        pdf_docs = self.load_pdf(str(pdf_file), document_name=doc_name)

                        # Update metadata to indicate ZIP source
                        for doc in pdf_docs:
                            doc.metadata["zip_name"] = zip_name
                            doc.metadata["document_name"] = doc_name
                            doc.metadata["source_type"] = "zip"

                        documents.extend(pdf_docs)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load PDF {pdf_file} from ZIP: {str(e)}"
                        )
                        continue

                logger.info(f"Loaded {len(documents)} documents from ZIP")
                return documents

        except Exception as e:
            logger.error(f"Failed to load ZIP {file_path}: {str(e)}")
            raise

    def load_document(
        self, file_path: str, document_name: str = None
    ) -> List[Document]:
        """
        Load document based on file extension.

        Args:
            file_path: Path to document file
            document_name: Optional document name

        Returns:
            List of Document objects

        Raises:
            ValueError: If file format is not supported
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".pdf":
            return self.load_pdf(file_path, document_name)
        elif file_ext == ".zip":
            return self.load_zip(file_path, document_name)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def validate_file_path(self, file_path: str) -> bool:
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: {path.suffix}")
            return False

        if path.stat().st_size == 0:
            logger.error(f"File is empty: {file_path}")
            return False

        logger.info(f"File validated: {file_path}")
        return True
