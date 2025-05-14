from .config import settings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

def get_vector_store(docs: list, rebuild: bool = False) -> FAISS:
    """
    Crea o carica indice FAISS per embeddings dei documenti.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    if rebuild or not os.path.exists(settings.VECTOR_STORE_PATH):
        store = FAISS.from_documents(docs, embeddings)
        store.save_local(settings.VECTOR_STORE_PATH)
    else:
        store = FAISS.load_local(settings.VECTOR_STORE_PATH, embeddings)
    return store