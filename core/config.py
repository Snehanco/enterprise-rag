from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys & URLs
    GROQ_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    # Collection Names
    COLLECTION_ORG: str = "org_knowledge"
    COLLECTION_USER: str = "user_uploads"

    # Models (FastEmbed supported)
    DENSE_MODEL: str = "BAAI/bge-small-en-v1.5"
    SPARSE_MODEL: str = "prithivida/Splade_PP_en_v1"
    COLBERT_MODEL: str = "colbert-ir/colbertv2.0"  # Standard ColBERT
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = ".env"


settings = Settings()
