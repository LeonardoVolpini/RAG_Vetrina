import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str = None
    VECTOR_STORE_PATH: str = "./vector_store.faiss"

    class Config:
        env_file = ".env"

settings = Settings()