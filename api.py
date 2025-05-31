from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from rag.generate import ingest_documents, ask_query
from rag.vector_store_singleton import VectorStoreSingleton
from rag.cache import ResponseCache
from rag.few_shot_examples import FewShotExampleManager
import os
import json
import requests

app = FastAPI()
response_cache = ResponseCache.get_instance(max_size=500, ttl=1800)  # Cache di 30 minuti

class CSVOptions(BaseModel):
    header_row: int = 0
    include_columns: Optional[List[str]] = None

class IngestRequest(BaseModel):
    file_paths: List[str]
    provider: Optional[str] = 'openai'  # 'openai', 'gemini', o 'llama'
    rebuild_index: bool = False
    callback_url: Optional[str] = None
    csv_options: Optional[CSVOptions] = None

class AskRequest(BaseModel):
    query: str
    provider: Optional[str] = 'openai'  # 'openai', 'gemini', o 'llama'
    model_name: Optional[str] = None
    use_cache: bool = False
    use_few_shot: bool = True  # Nuovo parametro per controllare few-shot
    max_examples: int = 3      # Numero massimo di esempi da utilizzare

class FewShotExample(BaseModel):
    question: str
    answer: str
    context_snapshot: Any
    reasoning: str

class AddExampleRequest(BaseModel):
    question: str
    answer: str
    context_snapshot: Any
    reasoning: str

class UpdateExampleRequest(BaseModel):
    index: int
    question: Optional[str] = None
    answer: Optional[str] = None
    context_snapshot: Optional[Any] = None
    reasoning: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Inizializzazione dello store al lancio dell'applicazione"""
    vector_store = VectorStoreSingleton.get_instance()
    vector_store.initialize()  # Default provider: openai

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
        vector_store.set_store(store, request.provider)
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
    
    if not vector_store.is_initialized():
        return {"status": "creating new index", "file_count": len(request.file_paths)}
    elif request.rebuild_index:
        return {"status": "rebuilding index", "file_count": len(request.file_paths)}
    else:
        return {"status": "adding to existing index", "file_count": len(request.file_paths)}

@app.post('/ask/')
async def ask(request: AskRequest):
    """Endpoint per eseguire una query RAG con supporto few-shot examples"""
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
        model = request.model_name or 'llama-model'

    use_few_shot = request.use_few_shot or True
    max_examples = request.max_examples or 3
    
    # Crea una chiave cache
    cache_key = f"{request.query}_{provider}_{model}_{use_few_shot}"
    
    # Verifica se la risposta è in cache
    if request.use_cache:
        cached_result = response_cache.get(cache_key, provider, model)
        if cached_result:
            return {**cached_result, "from_cache": True}
    
    # Ottieni lo store e esegui la query
    store = vector_store.get_store()
    
    try:
        result = ask_query(request.query, store, provider, model,
                            use_few_shot, max_examples)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Errore durante la query: {error_trace}")
        result = {"error": str(e), "answer": "Si è verificato un errore durante la generazione della risposta."}
    
    # Salva in cache
    if request.use_cache and "error" not in result:
        response_cache.set(cache_key, provider, model, result)
    
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
        "current_provider": vector_store.get_provider(),
        "few_shot_examples_enabled": True
    }
    
    # Se lo store è inizializzato, aggiungi dettagli
    if store:
        doc_count = len(store.index_to_docstore_id) if hasattr(store, 'index_to_docstore_id') else "unknown"
        info["document_count"] = doc_count
        info["status"] = "ready"
    else:
        info["document_count"] = 0
        info["status"] = "not_ready" if vector_store.is_initialized() else "needs_ingest"
    
    # Aggiungi informazioni sui few-shot examples
    try:
        example_manager = FewShotExampleManager()
        info["few_shot_examples_count"] = len(example_manager.get_examples())
    except Exception as e:
        info["few_shot_examples_count"] = 0
        info["few_shot_examples_error"] = str(e)
    
    # Carica metadata se disponibili
    try:
        from rag.config import settings
        metadata_path = settings.VECTOR_STORE_PATH + "_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                info["last_ingest"] = metadata
    except:
        pass
    
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
        return {
            "provider": provider, 
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        }


# --- ENDPOINT PER FEW-SHOT EXAMPLES ---

@app.get('/few-shot/examples/')
async def get_few_shot_examples():
    """Ottieni tutti gli esempi few-shot"""
    try:
        example_manager = FewShotExampleManager()
        examples = example_manager.get_examples()
        return {
            "total_examples": len(examples),
            "examples": examples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero degli esempi: {str(e)}")

@app.post('/few-shot/examples/')
async def add_few_shot_example(request: AddExampleRequest):
    """Aggiungi un nuovo esempio few-shot (solo question-answer)"""
    try:
        example_manager = FewShotExampleManager()
        
        # Se context_snapshot è un array di oggetti, trasformalo in stringa JSON
        context_snapshot = request.context_snapshot
        if isinstance(context_snapshot, list) or isinstance(context_snapshot, dict):
            context_snapshot = json.dumps(context_snapshot, ensure_ascii=False)
            
        example_manager.add_example(
            question=request.question,
            answer=request.answer,
            context_snapshot=context_snapshot,
            reasoning=request.reasoning
        )
        total_examples = len(example_manager.get_examples())
        return {
            "message": "Esempio aggiunto con successo",
            "total_examples": total_examples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'aggiunta dell'esempio: {str(e)}")

@app.put('/few-shot/examples/{example_index}')
async def update_few_shot_example(example_index: int, request: UpdateExampleRequest):
    """Aggiorna un esempio few-shot esistente"""
    try:
        example_manager = FewShotExampleManager()
        examples = example_manager.get_examples()
        
        if example_index < 0 or example_index >= len(examples):
            raise HTTPException(status_code=404, detail=f"Esempio con indice {example_index} non trovato")
        
        # Aggiorna solo i campi specificati
        if request.question is not None:
            examples[example_index]["question"] = request.question
        if request.answer is not None:
            examples[example_index]["answer"] = request.answer
        if request.context_snapshot is not None:
            examples[example_index]['context_snapshot'] = request.context_snapshot
        if request.reasoning is not None:
            examples[example_index]["reasoning"] = request.reasoning
        
        # Salva gli esempi aggiornati
        example_manager.examples = examples
        example_manager._save_examples(examples)
        
        return {
            "message": "Esempio aggiornato con successo",
            "updated_example": examples[example_index]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'aggiornamento dell'esempio: {str(e)}")

@app.delete('/few-shot/examples/{example_index}')
async def delete_few_shot_example(example_index: int):
    """Elimina un esempio few-shot"""
    try:
        example_manager = FewShotExampleManager()
        success = example_manager.remove_example(example_index)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Esempio con indice {example_index} non trovato")
        
        total_examples = len(example_manager.get_examples())
        return {
            "message": "Esempio eliminato con successo",
            "total_examples": total_examples
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'eliminazione dell'esempio: {str(e)}")

@app.get('/few-shot/examples/{example_index}')
async def get_few_shot_example(example_index: int):
    """Ottieni un singolo esempio few-shot per indice"""
    try:
        example_manager = FewShotExampleManager()
        examples = example_manager.get_examples()
        
        if example_index < 0 or example_index >= len(examples):
            raise HTTPException(status_code=404, detail=f"Esempio con indice {example_index} non trovato")
        
        return {
            "index": example_index,
            "example": examples[example_index]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero dell'esempio: {str(e)}")

@app.post('/few-shot/examples/bulk')
async def add_bulk_few_shot_examples(examples: List[FewShotExample]):
    """Aggiungi più esempi few-shot in una volta (solo question-answer)"""
    try:
        example_manager = FewShotExampleManager()
        added_count = 0
        
        for example in examples:
            example_manager.add_example(
                question=example.question,
                answer=example.answer,
                context_snapshot=example.context_snapshot,
                reasoning=example.reasoning
            )
            added_count += 1
        
        total_examples = len(example_manager.get_examples())
        return {
            "message": f"{added_count} esempi aggiunti con successo",
            "added_examples": added_count,
            "total_examples": total_examples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nell'aggiunta degli esempi: {str(e)}")

@app.get('/few-shot/preview/')
async def preview_few_shot_prompt(max_examples: int = 3, query: str = "esempio di query"):
    """Anteprima di come appaiono gli esempi nel prompt per una query specifica"""
    try:
        vector_store = VectorStoreSingleton.get_instance()
        store = vector_store.get_store()
        
        if not store:
            return {"error": "Vector store non inizializzato"}
        
        example_manager = FewShotExampleManager()
        formatted_examples = example_manager.get_relevant_examples(
            query=query,
            store=store,
            max_examples=max_examples
        )
        
        return {
            "query": query,
            "max_examples": max_examples,
            "formatted_prompt": formatted_examples,
            "total_available_examples": len(example_manager.get_examples())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nella generazione dell'anteprima: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    #Lunch with: uvicorn api:app --reload --port 8000