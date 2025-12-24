from pydantic import BaseModel, Field
from typing import List, Optional


class TreatmentPlanRequest(BaseModel):
    query: str = Field(..., description="The medical question or instruction")
    patient_id: str = Field(
        ..., description="Unique ID for the patient's diagnostic report"
    )
    disease_name: str = Field(
        ..., description="Name of the disease for protocol matching"
    )


class PatientSearchRequest(BaseModel):
    query: str = Field(..., description="The search query related to patient history")
    patient_id: str = Field(..., description="The document name or ID of the patient")


class ProtocolSearchRequest(BaseModel):
    query: str = Field(..., description="The search query related to medical protocols")
    disease_name: str = Field(
        ..., description="The document name or ID of the protocol"
    )


class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float
    page_number: Optional[int] = None


class SearchResponse(BaseModel):
    answer: str
    # Detailed evidence used for the answer
    relevant_chunks: List[RetrievedChunk]
    # Unique list of document names for a quick overview
    sources: List[str]
