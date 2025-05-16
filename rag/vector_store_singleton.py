from langchain_community.vectorstores import FAISS
import threading
import os
from .config import settings

class VectorStoreSingleton:
    _instance = None
    _lock = threading.Lock()
    _store = None
    _provider = None
    _is_initializing = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def initialize(self, provider='gemini', rebuild=False):
        """Inizializza lo store in modo thread-safe"""
        with self._lock:
            if self._is_initializing:
                return False
            
            # Validazione provider
            if provider not in ['openai', 'gemini', 'llama']:
                raise ValueError(f"Provider non supportato: {provider}")
            
            self._is_initializing = True
            self._provider = provider
            
            try:
                from .embeddings import get_embeddings
                embeddings = get_embeddings(provider)
                
                if not rebuild and os.path.exists(settings.VECTOR_STORE_PATH):
                    self._store = FAISS.load_local(
                        settings.VECTOR_STORE_PATH,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                else:
                    self._store = None
                
                return True
            finally:
                self._is_initializing = False
    
    def get_store(self):
        """Restituisce lo store corrente"""
        return self._store
    
    def set_store(self, store):
        """Imposta un nuovo store (dopo ingest)"""
        with self._lock:
            self._store = store
    
    def is_initialized(self):
        """Verifica se lo store è stato inizializzato"""
        return self._store is not None
        
    def get_provider(self):
        """Restituisce il provider corrente"""
        return self._provider