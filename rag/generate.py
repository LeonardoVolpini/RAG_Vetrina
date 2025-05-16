from .loader_pdf import load_pdfs
from .loader_csv import load_csv
from .embeddings import get_embeddings
from .retrieval import build_rag_chain
from .config import settings
import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS

def get_file_type(file_path: str) -> str:
    """Determina il tipo di file in base all'estensione"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.csv', '.xlsx', '.xls']:
        return 'tabular'
    else:
        # Default a PDF, ma potresti lanciare un'eccezione
        return 'unknown'

def ingest_documents(file_paths: list[str], rebuild_index: bool = False, provider: str = 'openai',
                     csv_options: Optional[Dict[str, Any]] = None) -> Any:
    """
    Funzione unificata per ingest: carica PDF e CSV, crea/carica indice FAISS e restituisce store.
    Supporta sia la ricostruzione completa che l'aggiunta incrementale di documenti.
    
    Args:
        file_paths: Lista di percorsi ai file
        rebuild_index: Se True, ricostruisce l'indice anche se esiste
        provider: Provider di embeddings ('openai' o 'gemini')
        csv_options: Opzioni specifiche per i file CSV (header_row, include_columns)
    """
    docs = []
    
    # Default opzioni CSV
    csv_options = csv_options or {}
    
    # Raggruppa i file per tipo
    for path in file_paths:
        file_type = get_file_type(path)
        
        if file_type == 'pdf':
            # Carica e aggiungi documenti PDF
            pdf_docs = load_pdfs([path])
            docs.extend(pdf_docs)
            
        elif file_type == 'tabular':
            # Carica e aggiungi documenti CSV/Excel
            csv_docs = load_csv(
                path, 
                header_row=csv_options.get('header_row', 0),
                include_columns=csv_options.get('include_columns')
            )
            docs.extend(csv_docs)
            
        elif file_type == 'unknown':
            print(f"Tipo di file non supportato: {path}")
    
    # Ottieni gli embeddings una sola volta per riuso
    embeddings = get_embeddings(provider)
    
    # Controlla se ci sono documenti da elaborare
    if not docs:
        raise ValueError("Nessun documento valido caricato per l'ingestione.")
    
    # Gestione dell'indice FAISS
    if rebuild_index or not os.path.exists(settings.VECTOR_STORE_PATH):
        # Crea un nuovo indice con i documenti caricati
        print(f"Creazione di un nuovo indice con {len(docs)} documenti")
        store = FAISS.from_documents(docs, embeddings)
        store.save_local(settings.VECTOR_STORE_PATH)
        return store
    else:
       # Aggiunta incrementale a un indice esistente
        print(f"Aggiunta di {len(docs)} nuovi documenti all'indice esistente")
        # Carica l'indice esistente con allow_dangerous_deserialization=True
        store = FAISS.load_local(
            settings.VECTOR_STORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Aggiungi i nuovi documenti all'indice
        store.add_documents(docs)
        
        # Salva l'indice aggiornato
        store.save_local(settings.VECTOR_STORE_PATH)
        return store

def ask_query(query: str, store, provider: str, model_name: str):
    """
    Esegue query RAG su indice già caricato con gestione errori.
    """
    try:
        rag_chain = build_rag_chain(store, provider, model_name)
        output = rag_chain.invoke(query)
        sources = [{
            "page": d.metadata.get('page', None),
            "source_pdf": d.metadata.get('source_pdf', None),
            "source_file": d.metadata.get('source_file', None),
            "row_index": d.metadata.get('row_index', None),
            "content_type": d.metadata.get('content_type', None)
        } for d in output.get('source_documents', [])]
        return {"answer": output.get('result'), "sources": sources}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Errore durante la query: {error_trace}")
        return {"error": str(e), "answer": "Si è verificato un errore durante la generazione della risposta."}