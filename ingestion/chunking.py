from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document


class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],  # Preserve semantic boundaries
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunked_docs = self.splitter.split_documents(documents)
        for i, doc in enumerate(chunked_docs):
            doc.metadata["chunk_index"] = i
        return chunked_docs
