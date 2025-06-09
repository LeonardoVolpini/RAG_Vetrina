from .loader_pdf import load_pdfs
from .loader_csv import load_csv
from .embeddings import get_embeddings
from .retrieval import build_rag_chain
from .config import settings
import os
from typing import Dict, Any, Optional, List
from langchain_community.vectorstores import Chroma 

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
        provider: Provider di embeddings ('openai', 'gemini', o 'llama')
        csv_options: Opzioni specifiche per i file CSV (header_row, include_columns)
    """
    # Validazione provider
    if provider not in ['openai', 'gemini', 'llama']:
        raise ValueError(f"Provider non supportato: {provider}")
    
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
    
    # Controlla se ci sono documenti da elaborare
    if not docs:
        raise ValueError("Nessun documento valido caricato per l'ingestione.")

    # Ottieni gli embeddings una sola volta per riuso
    embeddings = get_embeddings(provider)

    # Percorso per il database persistente di Chroma
    persist_directory = settings.VECTOR_STORE_PATH # Puoi riutilizzare lo stesso path

    if rebuild_index and os.path.exists(persist_directory):
        # Per Chroma, ricostruire significa cancellare la vecchia directory
        import shutil
        print(f"Ricostruzione richiesta, elimino la directory: {persist_directory}")
        shutil.rmtree(persist_directory)

    # Crea o carica il database Chroma
    # Chroma gestisce l'indicizzazione ANN automaticamente
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"Ingestione completata. {len(docs)} documenti aggiunti/aggiornati.")
    return store

def _extract_brand_filter(query: str, known_brands: List[str]) -> Optional[Dict[str, Any]]:
    """
    Estrae una marca conosciuta dalla query e crea un dizionario di filtro per ChromaDB.
    """
    for brand in known_brands:
        # Cerca la marca nella query (ignorando maiuscole/minuscole)
        if f' {brand.lower()} ' in f' {query.lower()} ':
            # Usa 'brand_normalized' creato in loader_csv.py
            filter_dict = {"brand_normalized": brand.lower().replace(' ', '_')}
            print(f"✅ Filtro per marca rilevato: {filter_dict}")
            return filter_dict
    return None

def ask_query(query: str, store, provider: str, model_name: str,
              use_few_shot: bool, max_examples: int,
              regenerateName: bool, generateDescription: bool):
    """
    Esegue query RAG, applicando filtri dinamici per la marca se rilevata nella query.
    
    Args:
        query: Query utente
        store: Vector store FAISS
        provider: Provider LLM ('openai', 'gemini', o 'llama')
        model_name: Nome del modello specifico
        use_few_shot: Se utilizzare few-shot examples
        max_examples: Numero massimo di esempi
        regenerateName: Flag per decidere se rigenare anche il nome
        generateDescription: Flag per decidere se generare la descrizione
    """
    # Validazione provider
    if provider not in ['openai', 'gemini', 'llama']:
        raise ValueError(f"Provider non supportato: {provider}")
        
    try:
        # Marche conosciute
        known_brands = [
                    "abk", "aco", "aeg", "albatros", "alubel",
                    "amonn", "appiani", "arblu", "bacchi", "bifire",
                    "bigmat", "boero", "bosch", "colorificiotirreno",
                    "cvr", "dakota", "delconca", "dewalt", "dorken",
                    "duramitt", "edilferro", "ediltec", "einhell",
                    "faraone", "fassabortolo", "fila", "firstcorporation",
                    "fischer", "fitt", "gattoni", "hidra", "hilti",
                    "icobit", "imer", "index", "isolmant", "isover",
                    "itwitaly", "kapriol" , "karcher", "kerakoll", "knauf",
                    "laticrete", "leca", "madras", "makita", "mapei",
                    "maurer", "novellini", "oikos", "olympia", "onduline",
                    "palazzetti", "papillon", "pastorelli", "poron",
                    "profiltec", "ragno", "raimondi", "sait", "saniplast",
                    "sanmarco", "schulz", "sika", "soprema", "spit",
                    "unifix", "unishop", "upower", "ursa", "volteco",
                    "weber", "yamato"                     
                    ]
        
        filter_dict = _extract_brand_filter(query, known_brands)
        
        # Costruisci i search_kwargs
        search_kwargs = {"k": 5, "score_threshold": 0.95}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        rag_chain = build_rag_chain(
            store, provider, model_name, use_few_shot, max_examples, 
            regenerateName, generateDescription,
            search_kwargs=search_kwargs
        )
        
        output = rag_chain.invoke(query)
        sources = [{
            "page": d.metadata.get('page', None),
            "source_pdf": d.metadata.get('source_pdf', None),
            "source_file": d.metadata.get('source_file', None),
            "row_index": d.metadata.get('row_index', None),
            "content_type": d.metadata.get('content_type', None),
            "brand": d.metadata.get('brand', 'N/D')
        } for d in output.get('source_documents', [])]
        
        return {
            "answer": output.get('result'), 
            "sources": sources
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Errore durante la query: {error_trace}")
        return {"error": str(e), "answer": "Si è verificato un errore durante la generazione della risposta."}