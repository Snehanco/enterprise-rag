import os
from langchain_groq import ChatGroq
from core.config import settings


class LLMClient:
    def __init__(self):
        # Ensure GROQ_API_KEY is in your .env or settings
        self.api_key = settings.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing from environment variables.")

        self.llm = ChatGroq(
            temperature=0.0, model_name=settings.LLM_MODEL, api_key=self.api_key
        )

    def get_llm(self):
        return self.llm
