import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI settings (opzionale se usi solo Gemini)
    OPENAI_API_KEY: str = None
    
    # Google Gemini settings (necessario per l'API gratuita)
    GEMINI_API_KEY: str = None
    
    # Llama settings
    LLAMA_API_KEY: str = None
    LLAMA_API_BASE: str = "https://api.cloud.llamaindex.ai"  # LlamaCloud URL
    
    # Vector store settings
    VECTOR_STORE_PATH: str = "./vector_store"

    class Config:
        env_file = ".env"

settings = Settings()