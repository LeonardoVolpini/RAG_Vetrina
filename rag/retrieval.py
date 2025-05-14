from langchain.chains import RetrievalQA
from .embeddings import get_vector_store
from .config import settings

# import LLM wrappers
from langchain_community.llms import OpenAI
from google import genai


def get_llm(provider: str, model_name: str):
    """
    Ritorna un oggetto LLM LangChain in base al provider scelto.
    provider: 'openai' o 'gemini'
    """
    if provider == 'openai':
        return OpenAI(model_name=model_name, openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        # google-genai client
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        # wrapper minimale per LangChain-like interface
        class GeminiLLM:
            def __init__(self, client, model):
                self.client = client
                self.model = model
            def __call__(self, prompt: str) -> str:
                response = self.client.generate_text(model=self.model, prompt=prompt)
                return response.text
        return GeminiLLM(client, model_name)
    else:
        raise ValueError("Provider LLM non valido")


def build_rag_chain(pdf_paths: list[str], provider: str = 'openai', model_name: str = 'gpt-3.5-turbo', rebuild_index: bool = False):
    """
    1. Carica PDF
    2. Costruisci/carica FAISS
    3. Costruisci RetrievalQA usando il provider scelto
    """
    from .loader import load_pdfs
    docs = load_pdfs(pdf_paths)
    store = get_vector_store(docs, rebuild=rebuild_index)
    retriever = store.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(provider, model_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )