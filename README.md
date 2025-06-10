# RAG_Vetrina
RAG (Retrieval Augmented Generation) per Vetrina, specializzato sui prodotti dei magazzini edili

Struttura:

'''
rag_project/
├── rag/
│   ├── __init__.py       # pacchetto Python
│   ├── config.py         # configurazioni (API keys, path)
│   ├── loader.py         # ingest PDF -> Documenti LangChain
│   ├── embeddings.py     # embeddings + Chroma
│   ├── retrieval.py      # costruzione del retriever + QA chain
│   └── generate.py        # generazione risposte RAG
└── api.py                # FastAPI esporre endpoint RAG per NodeJS (ingest + ask)
'''