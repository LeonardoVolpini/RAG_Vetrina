from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from rag.generate import ingest_documents, ask_query
from rag.vector_store_singleton import VectorStoreSingleton
from rag.cache import ResponseCache
import requests

app = FastAPI()
response_cache = ResponseCache.get_instance(max_size=500, ttl=1800)  # Cache di 30 minuti

class CSVOptions(BaseModel):
    header_row: int = 0
    include_columns: Optional[List[str]] = None

class IngestRequest(BaseModel):
    file_paths: List[str]
    provider: Optional[str] = 'openai'  # 'openai', 'gemini', o 'llama'
    model_name: Optional[str] = None
    rebuild_index: bool = False
    callback_url: Optional[str] = None
    csv_options: Optional[CSVOptions] = None

class AskRequest(BaseModel):
    query: str
    provider: Optional[str] = 'openai'  # 'openai', 'gemini', o 'llama'
    model_name: Optional[str] = None
    use_cache: bool = True

@app.on_event("startup")
async def startup_event():
    """Inizializzazione dello store al lancio dell'applicazione"""
    vector_store = VectorStoreSingleton.get_instance()
    vector_store.initialize()  # Default provider: gemini

@app.post('/ingest/')
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """Endpoint per l'ingest dei documenti (PDF e CSV/Excel)"""
    # Validazione del provider
    if request.provider not in ['openai', 'gemini', 'llama']:
        raise HTTPException(status_code=400, detail=f"Provider non supportato: {request.provider}")
    
    vector_store = VectorStoreSingleton.get_instance()
    
    # Funzione per eseguire l'ingest in background
    def do_ingest():
        csv_options_dict = request.csv_options.dict() if request.csv_options else {}
        store = ingest_documents(
            file_paths=request.file_paths,
            rebuild_index=request.rebuild_index,
            provider=request.provider,
            csv_options=csv_options_dict
        )
        vector_store.set_store(store)
        print("Ingestione completata:", len(request.file_paths), "documenti elaborati.")
        
        # Invia notifica al callback_url se specificato     TODO non funziona
        if request.callback_url:
            try:
                requests.post(
                    request.callback_url, 
                    json={"status": "complete", "file_count": len(request.file_paths)}
                )
            except Exception as e:
                print(f"Errore nell'invio della notifica callback: {str(e)}")
    
    # Esegui sempre in background, sia per rebuild che per aggiunta incrementale
    background_tasks.add_task(do_ingest)
    
    if not vector_store.is_initialized() :
        return {"status": "creating new index", "file_count": len(request.file_paths)}
    elif request.rebuild_index:
        return {"status": "rebuilding index", "file_count": len(request.file_paths)}
    else:
        return {"status": "adding to existing index", "file_count": len(request.file_paths)}

@app.post('/ask/')
async def ask(request: AskRequest):
    """Endpoint per eseguire una query RAG"""
    # Validazione del provider
    if request.provider not in ['openai', 'gemini', 'llama']:
        raise HTTPException(status_code=400, detail=f"Provider non supportato: {request.provider}")
    
    vector_store = VectorStoreSingleton.get_instance()
    
    if not vector_store.is_initialized():
        return {"error": "Indice non inizializzato. Chiamare prima /ingest/."}
    
    provider = request.provider
    
    # Determinazione del modello di default in base al provider
    if provider == 'openai':
        model = request.model_name or 'gpt-3.5-turbo'
    elif provider == 'gemini':
        model = request.model_name or 'gemini-models/gemini-1.5-pro-latest'
    elif provider == 'llama':
        model = request.model_name or 'llama-model'  # Modello di default per Llama
    
    # Verifica se la risposta è in cache
    if request.use_cache:
        cached_result = response_cache.get(request.query, provider, model)
        if cached_result:
            return {**cached_result, "from_cache": True}
    
    # Ottieni lo store e esegui la query
    store = vector_store.get_store()
    result = ask_query(request.query, store, provider, model)
    
    # Salva in cache
    if request.use_cache and "error" not in result:
        response_cache.set(request.query, provider, model, result)
    
    return result

@app.get('/info/')
async def get_info():
    """Endpoint per ottenere informazioni sul sistema RAG"""
    vector_store = VectorStoreSingleton.get_instance()
    store = vector_store.get_store()
    
    info = {
        "is_initialized": vector_store.is_initialized(),
        "supported_file_types": ["pdf", "csv", "xlsx", "xls"],
        "supported_providers": ["openai", "gemini", "llama"],
        "provider": vector_store._provider,
    }
    
    # Se lo store è inizializzato, aggiungi dettagli
    if store:
        # FAISS non espone questa informazione direttamente, quindi dobbiamo usare l'indice docstore interno
        doc_count = len(store.index_to_docstore_id) if hasattr(store, 'index_to_docstore_id') else "unknown"
        info["document_count"] = doc_count
    
    return info

@app.get('/models/{provider}')
async def get_models(provider: str):
    """Endpoint per ottenere i modelli disponibili per un provider specifico"""
    if provider not in ['openai', 'gemini', 'llama']:
        raise HTTPException(status_code=400, detail=f"Provider non supportato: {provider}")
    
    if provider == 'gemini':
        from rag.retrieval import supported_gemini_models
        models = supported_gemini_models()
        return {"provider": provider, "models": [m.name for m in models if "generateContent" in m.supported_generation_methods]}
    elif provider == 'llama':
        from rag.retrieval import supported_llama_models
        models = supported_llama_models()
        return {"provider": provider, "models": [m["id"] for m in models]}
    elif provider == 'openai':
        # Modelli fissi per OpenAI (potrebbe essere esteso con una chiamata API)
        return {
            "provider": provider, 
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)