from .config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

def get_embeddings(provider='openai'):
    """
    Restituisce un oggetto embeddings in base al provider scelto.
    Supporta OpenAI e Google Vertex AI (per Gemini).
    """
    if provider == 'openai':
        return OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        # Usa VertexAI Embeddings per Gemini
        return VertexAIEmbeddings(
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION,
            credentials=settings.GOOGLE_CREDENTIALS
        )
    else:
        raise ValueError(f"Provider embedding non supportato: {provider}")

def get_vector_store(docs: list, rebuild: bool = False, provider='openai') -> FAISS:
    """
    Crea o carica indice FAISS per embeddings dei documenti.
    """
    embeddings = get_embeddings(provider)
    if rebuild or not os.path.exists(settings.VECTOR_STORE_PATH):
        store = FAISS.from_documents(docs, embeddings)
        store.save_local(settings.VECTOR_STORE_PATH)
    else:
        store = FAISS.load_local(settings.VECTOR_STORE_PATH, embeddings)
    return store