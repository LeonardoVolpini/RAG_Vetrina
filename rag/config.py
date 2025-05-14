import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI settings
    OPENAI_API_KEY: str
    
    # Google Gemini settings
    GEMINI_API_KEY: str = None
    
    # Google Vertex AI settings
    GOOGLE_PROJECT_ID: str = None
    GOOGLE_LOCATION: str = "us-central1"  # Default region
    GOOGLE_CREDENTIALS: str = None  # Path to Google credentials JSON file
    
    # Vector store settings
    VECTOR_STORE_PATH: str = "./vector_store.faiss"

    class Config:
        env_file = ".env"

settings = Settings()