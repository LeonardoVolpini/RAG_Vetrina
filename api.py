from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from rag.generate import ingest_pdfs, ask_query

app = FastAPI()
_store = None
_current_provider = 'openai'
_current_model = 'gpt-3.5-turbo'

class IngestRequest(BaseModel):
    pdf_paths: List[str]
    provider: Optional[str] = 'openai'
    model_name: Optional[str] = None
    rebuild_index: bool = False

class AskRequest(BaseModel):
    query: str

@app.post('/ingest/')
async def ingest(request: IngestRequest):
    global _store, _current_provider, _current_model
    _current_provider = request.provider
    _current_model = request.model_name or ('gemini-models/gemini-1.5-pro-latest' if request.provider == 'gemini' else 'gpt-3.5-turbo')
    _store = ingest_pdfs(request.pdf_paths, rebuild_index=request.rebuild_index, provider=_current_provider)
    return {"status": "index built", "pdf_count": len(request.pdf_paths)}

@app.post('/ask/')
async def ask(request: AskRequest):
    if _store is None:
        return {"error": "Indice non inizializzato. Chiamare prima /ingest/."}
    result = ask_query(request.query, _store, _current_provider, _current_model)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)