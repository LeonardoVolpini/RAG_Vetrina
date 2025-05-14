from langchain.chains import RetrievalQA
from .config import settings

# import LLM wrappers
from langchain_openai import OpenAI
import google.generativeai as genai

def get_llm(provider: str, model_name: str):
    """
    Ritorna un oggetto LLM LangChain in base al provider scelto.
    provider: 'openai' o 'gemini'
    """
    if provider == 'openai':
        return OpenAI(model_name=model_name, openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        # Configura l'API Gemini con la chiave
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Wrapper per l'interfaccia LangChain
        class GeminiLLM:
            def __init__(self, model_name="gemini-pro"):
                self.model = genai.GenerativeModel(model_name)
                
            def __call__(self, prompt: str) -> str:
                try:
                    response = self.model.generate_content(prompt)
                    # Gestisce diversi tipi di risposta possibili
                    if hasattr(response, 'text'):
                        return response.text
                    elif hasattr(response, 'parts'):
                        return ''.join(part.text for part in response.parts)
                    else:
                        return str(response)
                except Exception as e:
                    print(f"Errore nella generazione con Gemini: {e}")
                    return f"Errore nel processare la richiesta: {str(e)}"
        
        return GeminiLLM(model_name=model_name or "gemini-pro")
    else:
        raise ValueError(f"Provider LLM non valido: {provider}")


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