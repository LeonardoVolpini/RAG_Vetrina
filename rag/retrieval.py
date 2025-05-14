from langchain.chains import RetrievalQA
from .config import settings

# import LLM wrappers
from langchain_openai import OpenAI
from langchain_google_vertexai import VertexAI
import google.generativeai as genai
import os

def get_llm(provider: str, model_name: str):
    """
    Ritorna un oggetto LLM LangChain in base al provider scelto.
    provider: 'openai' o 'gemini'
    """
    if provider == 'openai':
        return OpenAI(model_name=model_name, openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        if settings.GOOGLE_CREDENTIALS:
            # Usa Vertex AI se sono disponibili le credenziali complete
            return VertexAI(
                model_name=model_name,
                project=settings.GOOGLE_PROJECT_ID,
                location=settings.GOOGLE_LOCATION,
                credentials=settings.GOOGLE_CREDENTIALS
            )
        else:
            # Usa l'API Gemini con solo API key
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # wrapper minimale per LangChain-like interface
            class GeminiLLM:
                def __init__(self, model_name):
                    self.model = genai.GenerativeModel(model_name)
                
                def __call__(self, prompt: str) -> str:
                    response = self.model.generate_content(prompt)
                    return response.text
            
            return GeminiLLM(model_name)
    else:
        raise ValueError("Provider LLM non valido")


def build_rag_chain(store, provider: str = 'openai', model_name: str = 'gpt-3.5-turbo'):
    """
    Costruisce un RetrievalQA chain usando il vectorstore fornito e il provider scelto
    """
    retriever = store.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(provider, model_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )