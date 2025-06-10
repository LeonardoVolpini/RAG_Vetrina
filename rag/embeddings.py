from .config import settings
from langchain_openai import OpenAIEmbeddings
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
        self.api_base = api_base or settings.LLAMA_API_BASE
        # Use the correct embedding model for LlamaCloud
        self.model = "llama-text-embeddings"  # Modificato: usando modello appropriato per LlamaCloud
        
    def embed_documents(self, texts):
        """Genera embedding per una lista di testi"""
        embeddings = []
        for text in texts:
            try:
                # Utilizziamo formato per LlamaCloud
                response = requests.post(
                    f"{self.api_base}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"input": text, "model": self.model}
                )
                response.raise_for_status()  # Raise exception for HTTP errors
                result = response.json()
                
                # Log the response structure to debug
                print(f"LlamaCloud embedding response structure: {result.keys()}")
                
                # Extract embedding based on actual response structure
                if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
                    embeddings.append(result["data"][0]["embedding"])
                else:
                    raise ValueError(f"Unexpected response structure from LlamaCloud: {result}")
                    
            except Exception as e:
                error_message = str(e)
                response_text = response.text if 'response' in locals() else 'No response'
                print(f"Error from LlamaCloud API: {error_message}")
                print(f"Response text: {response_text}")
                raise ValueError(f"Error from LlamaCloud API: {error_message} - Response: {response_text}")
                
        return embeddings
        
    def embed_query(self, text):
        """Genera embedding per una query singola"""
        try:
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"input": text, "model": self.model}
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            result = response.json()
            
            # Extract embedding based on actual response structure
            if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
                return result["data"][0]["embedding"]
            else:
                raise ValueError(f"Unexpected response structure from LlamaCloud: {result}")
                
        except Exception as e:
            error_message = str(e)
            response_text = response.text if 'response' in locals() else 'No response'
            print(f"Error from LlamaCloud API: {error_message}")
            print(f"Response text: {response_text}")
            raise ValueError(f"Error from LlamaCloud API: {error_message} - Response: {response_text}")

def get_embeddings(provider='openai'):
    """
    Restituisce un oggetto embeddings in base al provider scelto.
    Supporta OpenAI, Google Generative AI (per Gemini) e Llama.
    """
    if provider == 'openai':
        return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        return GeminiEmbeddings()
    elif provider == 'llama':
        return LlamaEmbeddings()
    else:
        raise ValueError(f"Provider embedding non supportato: {provider}")