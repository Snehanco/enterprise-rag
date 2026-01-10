import logging
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from core.config import settings
from core.vectorstore import QdrantManager
from execution.chains import MedicalChainFactory
from execution.schemas import (
    PatientSearchRequest,
    ProtocolSearchRequest,
    RetrievedChunk,
    SearchResponse,
    TreatmentPlanRequest,
)
from ingestion.embeddings_manager import EmbeddingsManager
from execution.llm_client import LLMClient

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.qdrant = QdrantManager()
        self.embedder = EmbeddingsManager()
        self.llm = LLMClient().get_llm()
        # Initialize specialized chains
        self.patient_chain = MedicalChainFactory.get_patient_analysis_chain(self.llm)
        self.protocol_chain = MedicalChainFactory.get_protocol_explanation_chain(
            self.llm
        )
        self.plan_chain = MedicalChainFactory.get_treatment_plan_chain(self.llm)

    def _format_docs(self, docs) -> str:
        """Formats retrieved documents for the LLM context."""
        if not docs:
            return "No relevant documents found."

        formatted = []
        for doc in docs:
            payload = doc.payload or {}
            source = payload.get("document_name", "Unknown")
            content = payload.get("page_content", "")
            # We can also log the score for debugging
            score = round(doc.score, 4)
            formatted.append(f"SOURCE: {source} (Score: {score})\nCONTENT: {content}\n")
        return "\n---\n".join(formatted)

    def _process_results(self, docs: List[Any], answer: str) -> SearchResponse:
        """
        Helper to transform raw Qdrant points and LLM answer
        into a structured SearchResponse.
        """
        relevant_chunks = []
        unique_sources = set()

        for d in docs:
            payload = d.payload or {}
            source_name = payload.get("document_name", "Unknown")
            file_name = (
                payload.get("file_name") or payload.get("source") or "Unknown File"
            )

            # Create the chunk object with its score
            chunk = RetrievedChunk(
                content=payload.get("page_content", ""),
                source=source_name,
                score=round(d.score, 4),  # Direct ColBERT/MaxSim score
                page_number=payload.get("page_number"),
                file_name=file_name,
            )
            relevant_chunks.append(chunk)
            unique_sources.add(source_name)

        return SearchResponse(
            answer=answer, relevant_chunks=relevant_chunks, sources=list(unique_sources)
        )

    def generate_treatment_plan(self, request: TreatmentPlanRequest) -> SearchResponse:
        logger.info(
            f"FORCED SYNC: Fetching both {request.patient_id} and {request.disease_name}"
        )

        query_vecs = self.embedder.embed_query(request.query)

        # 1. INDEPENDENT SEARCHES (The "Guardrail")
        # We force the system to get the best patient chunks separately
        patient_docs = self.qdrant.search_single_collection(
            collection_name=settings.COLLECTION_USER,
            query_embeddings=query_vecs,
            search_filter=self.qdrant.create_match_filter(
                "document_name", request.patient_id
            ),
            limit=5,
        )

        # We force the system to get the best protocol chunks separately
        protocol_docs = self.qdrant.search_single_collection(
            collection_name=settings.COLLECTION_ORG,
            query_embeddings=query_vecs,
            search_filter=self.qdrant.create_match_filter(
                "document_name", request.disease_name
            ),
            limit=7,
        )

        # 2. Safety Check: Raise error/log if patient data is actually missing
        if not patient_docs:
            logger.error(
                f"CRITICAL: No patient data found for ID: {request.patient_id}"
            )
            # We still proceed, but the LLM will be warned

        # 3. CONTEXT SEPARATION (The "Architecture")
        # Instead of one big list, we split them so the LLM CANNOT miss them
        patient_context = "\n".join(
            [
                f"- [PAGE {d.payload.get('page_number', 'N/A')}]: {d.payload.get('page_content')}"
                for d in patient_docs
            ]
        )

        protocol_context = "\n".join(
            [
                f"- [PAGE {d.payload.get('page_number', 'N/A')}]: {d.payload.get('page_content')}"
                for d in protocol_docs
            ]
        )

        # 4. Invoke LLM with explicit sectioning
        # We pass these as separate variables to the prompt
        logger.info("Invoking LLM with dual-source context...")
        answer = self.plan_chain.invoke(
            {
                "patient_data": (
                    patient_context
                    if patient_context
                    else "NO PATIENT DATA FOUND IN DATABASE"
                ),
                "protocol_data": (
                    protocol_context
                    if protocol_context
                    else "NO PROTOCOL DATA FOUND IN DATABASE"
                ),
                "question": request.query,
            }
        )

        # 5. Combine docs for the JSON response metadata
        combined_docs = patient_docs + protocol_docs
        return self._process_results(combined_docs, answer)

    def search_patient_records(self, request: PatientSearchRequest) -> SearchResponse:
        logger.info(f"Analyzing patient records for: {request.patient_id}")

        query_vecs = self.embedder.embed_query(request.query)
        target = [
            {
                "collection": settings.COLLECTION_USER,
                "filter": self.qdrant.create_match_filter(
                    "document_name", request.patient_id
                ),
            }
        ]

        docs = self.qdrant.search_multiple_collections(query_vecs, target, limit=8)

        context_text = "\n\n".join([d.payload.get("page_content", "") for d in docs])
        answer = self.patient_chain.invoke(
            {"context": context_text, "question": request.query}
        )

        return self._process_results(docs, answer)

    def search_medical_protocols(
        self, request: ProtocolSearchRequest
    ) -> SearchResponse:
        logger.info(f"Retrieving protocols for: {request.disease_name}")

        query_vecs = self.embedder.embed_query(request.query)
        target = [
            {
                "collection": settings.COLLECTION_ORG,
                "filter": self.qdrant.create_match_filter(
                    "document_name", request.disease_name
                ),
            }
        ]

        docs = self.qdrant.search_multiple_collections(query_vecs, target, limit=8)

        context_text = "\n\n".join([d.payload.get("page_content", "") for d in docs])
        answer = self.protocol_chain.invoke(
            {"context": context_text, "question": request.query}
        )

        return self._process_results(docs, answer)
