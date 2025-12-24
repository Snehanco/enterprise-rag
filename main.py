import logging
import os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException
from core.vectorstore import QdrantManager
from execution.rag_service import RAGService
from execution.schemas import (
    PatientSearchRequest,
    ProtocolSearchRequest,
    SearchResponse,
    TreatmentPlanRequest,
)
from ingestion.pipeline import IngestionPipeline
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 1. Lifespan for Startup/Shutdown Logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize collections if they don't exist
    logger.info("Checking Qdrant collections...")
    try:
        qdrant_manager = QdrantManager()
        qdrant_manager.initialize_collections()
        print("Qdrant initialized successfully.")
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")

    yield  # The app runs here

    print("Shutting down...")


app = FastAPI(title="RAG Hybrid Search API", lifespan=lifespan)
ingestor = IngestionPipeline()


# 2. Ingestion Endpoint
@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    document_name: str = Form(...),
):
    """
    Uploads a file, saves it temporarily, and triggers the ingestion pipeline.
    """
    # Create a temporary path to save the uploaded file
    temp_path = f"{file.filename}"

    try:
        # Save file to disk
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Call your ingestion pipeline
        # Assuming process_document is a method in IngestionPipeline
        result = ingestor.process_document(
            file_path=temp_path,
            document_name=document_name,
            collection_name=collection_name,
        )

        return {
            "status": "success",
            "message": f"Document '{document_name}' ingested into '{collection_name}'",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_rag_service():
    return RAGService()


@app.post("/search/patient")
async def search_patient(
    request: PatientSearchRequest, service: RAGService = Depends(get_rag_service)
):
    try:
        response = service.search_patient_records(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/protocol")
async def search_protocol(
    request: ProtocolSearchRequest, service: RAGService = Depends(get_rag_service)
):
    try:
        response = service.search_medical_protocols(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/treatment-plan", response_model=SearchResponse)
async def create_plan(
    request: TreatmentPlanRequest, service: RAGService = Depends(get_rag_service)
):
    try:
        return service.generate_treatment_plan(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point"""

    # Get API configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    # Start server
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"API Documentation: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
