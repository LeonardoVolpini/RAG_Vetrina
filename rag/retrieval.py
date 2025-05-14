from langchain.chains import RetrievalQA
from .config import settings

# import LLM wrappers
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(provider: str, model_name: str):
    """
    Ritorna un oggetto LLM LangChain in base al provider scelto.
    provider: 'openai' o 'gemini'
    """
    if provider == 'openai':
        return OpenAI(model_name=model_name, openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        # Usa l'integrazione ufficiale di LangChain per Google Generative AI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-pro",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.7,
        )
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