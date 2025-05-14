from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from rag.generate import ingest_pdfs, ask_query
from rag.vector_store_singleton import VectorStoreSingleton
from rag.cache import ResponseCache
import requests

app = FastAPI()
response_cache = ResponseCache.get_instance(max_size=500, ttl=1800)  # Cache di 30 minuti

class IngestRequest(BaseModel):
    pdf_paths: List[str]
    provider: Optional[str] = 'openai'
    model_name: Optional[str] = None
    rebuild_index: bool = False
    callback_url: Optional[str] = None

class AskRequest(BaseModel):
    query: str
    provider: Optional[str] = 'openai'
    model_name: Optional[str] = None
    use_cache: bool = True

@app.on_event("startup")
async def startup_event():
    """Inizializzazione dello store al lancio dell'applicazione"""
    vector_store = VectorStoreSingleton.get_instance()
    vector_store.initialize()  # Default provider: gemini

@app.post('/ingest/')
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """Endpoint per l'ingest dei PDF"""
    vector_store = VectorStoreSingleton.get_instance()
    
    # Funzione per eseguire l'ingest in background
    def do_ingest():
        store = ingest_pdfs(request.pdf_paths,
                            rebuild_index=request.rebuild_index,
                            provider=request.provider)
        vector_store.set_store(store)
        print("Ingestione completata:", len(request.pdf_paths), "PDF elaborati.")   # TODO: avvisare utente via callback_url
    
    # Se rebuild o non inizializzato, esegui in background
    if request.rebuild_index or not vector_store.is_initialized():
        background_tasks.add_task(do_ingest)
        return {"status": "ingestion started", "pdf_count": len(request.pdf_paths)}
    else:
        return {"status": "index already exists", "message": "Use rebuild_index=true to rebuild"}

@app.post('/ask/')
async def ask(request: AskRequest):
    """Endpoint per eseguire una query RAG"""
    vector_store = VectorStoreSingleton.get_instance()
    
    if not vector_store.is_initialized():
        return {"error": "Indice non inizializzato. Chiamare prima /ingest/."}
    
    provider = request.provider
    model = request.model_name or ('gemini-models/gemini-1.5-pro-latest' if provider == 'gemini' else 'gpt-3.5-turbo')
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)