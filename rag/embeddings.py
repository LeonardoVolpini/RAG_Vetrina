from .config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
import os
import requests

class GeminiEmbeddings(Embeddings):
    """Classe personalizzata per utilizzare gli embeddings di Google Generative AI"""
    
    def __init__(self, api_key=None):
        """Inizializza con l'API key di Gemini"""
        self.api_key = api_key or settings.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        
    def embed_documents(self, texts):
        """Genera embedding per una lista di testi"""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings
        
    def embed_query(self, text):
        """Genera embedding per una query singola"""
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]

class LlamaEmbeddings(Embeddings):
    """Classe per utilizzare gli embeddings di LlamaCloud"""
    
    def __init__(self, api_key=None, api_base=None):
        """Inizializza con l'API key e base URL di Llama"""
        self.api_key = api_key or settings.LLAMA_API_KEY
        self.api_base = "https://api.cloud.llamaindex.ai/v1" # Endpoint LlamaCloud
        self.model = "text-embedding-ada-002"  # Modello compatibile con OpenAI
        
    def embed_documents(self, texts):
        """Genera embedding per una lista di testi"""
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": text, "model": self.model}
            )
            if response.status_code == 200:
                result = response.json()
                embeddings.append(result["data"][0]["embedding"])
            else:
                raise ValueError(f"Error from LlamaCloud API: {response.text}")
        return embeddings
        
    def embed_query(self, text):
        """Genera embedding per una query singola"""
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"input": text, "model": self.model}
        )
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            raise ValueError(f"Error from LlamaCloud API: {response.text}")

def get_embeddings(provider='openai'):
    """
    Restituisce un oggetto embeddings in base al provider scelto.
    Supporta OpenAI, Google Generative AI (per Gemini) e Llama.
    """
    if provider == 'openai':
        return OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        return GeminiEmbeddings()
    elif provider == 'llama':
        return LlamaEmbeddings()
    else:
        raise ValueError(f"Provider embedding non supportato: {provider}")

# Funzione attualmente inutilizzata
def get_vector_store(docs: list, rebuild: bool = False, provider='openai') -> FAISS:
    """
    Crea o carica indice FAISS per embeddings dei documenti.
    Supporta sia ricostruzione completa che aggiunta incrementale.
    """
    embeddings = get_embeddings(provider)
    
    if rebuild or not os.path.exists(settings.VECTOR_STORE_PATH):
        # Crea un nuovo indice
        store = FAISS.from_documents(docs, embeddings)
        store.save_local(settings.VECTOR_STORE_PATH)
    else:
        # Carica l'indice esistente con allow_dangerous_deserialization=True
        store = FAISS.load_local(
            settings.VECTOR_STORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True  # Aggiunto questo parametro
        )
        
        # Aggiungi i nuovi documenti se ce ne sono
        if docs:
            store.add_documents(docs)
            store.save_local(settings.VECTOR_STORE_PATH)
            
    return store