import logging
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

logger = logging.getLogger(__name__)


class MedicalChainFactory:
    @staticmethod
    def get_patient_analysis_chain(llm):
        """Chain for interpreting patient-specific diagnostic data."""
        prompt = ChatPromptTemplate.from_template(
            """
            SYSTEM: You are a Medical Diagnostic Assistant. Your task is to summarize and analyze 
            patient-specific records. Focus on clinical findings, lab results, and history.
            
            CONTEXT:
            {context}
            
            USER QUESTION: {question}
            
            ANALYSIS:
            """
        )
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    @staticmethod
    def get_protocol_explanation_chain(llm):
        """Chain for explaining organizational medical protocols."""
        prompt = ChatPromptTemplate.from_template(
            """
            SYSTEM: You are a Medical Protocol Expert. Explain the standard operating procedures
            and treatment guidelines based ONLY on the provided institutional knowledge.
            
            CONTEXT:
            {context}
            
            USER QUESTION: {question}
            
            GUIDELINE SUMMARY:
            """
        )
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    @staticmethod
    def get_treatment_plan_chain(llm):
        prompt = ChatPromptTemplate.from_template(
            """
            SYSTEM: You are a Clinical AI. You MUST use both the Patient History and the Medical Protocols provided.
            
            SECTION 1: PATIENT SPECIFIC HISTORY
            {patient_data}
            
            SECTION 2: MEDICAL PROTOCOLS & GUIDELINES
            {protocol_data}
            
            USER REQUEST: {question}
            
            INSTRUCTIONS:
            1. Start by summarizing the patient's specific condition from Section 1.
            2. Apply the guidelines from Section 2 to this specific patient.
            3. Cite the Page Numbers for every claim you make.
            
            TREATMENT PLAN:
            """
        )
        # Note: The keys here must match the dictionary keys in rag_service.py
        return RunnablePassthrough() | prompt | llm | StrOutputParser()
