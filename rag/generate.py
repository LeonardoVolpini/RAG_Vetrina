from .loader import load_pdfs
from .embeddings import get_vector_store
from .retrieval import build_rag_chain


def ingest_pdfs(pdf_paths: list[str], rebuild_index: bool = False, provider: str = 'openai'):
    """
    Funzione per ingest: carica PDF, crea/carica indice FAISS e restituisce store e chain.
    """
    docs = load_pdfs(pdf_paths)
    store = get_vector_store(docs, rebuild=rebuild_index, provider=provider)
    return store


def ask_query(query: str, store, provider: str, model_name: str):
    """
    Esegue query RAG su indice già caricato.
    """
    rag_chain = build_rag_chain(store, provider, model_name)
    output = rag_chain.invoke(query)
    sources = [{
        "page": d.metadata.get('page'),
        "source_pdf": d.metadata.get('source_pdf')
    } for d in output.get('source_documents', [])]
    return {"answer": output.get('result'), "sources": sources}