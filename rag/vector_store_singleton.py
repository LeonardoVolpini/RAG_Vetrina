from langchain_chroma import Chroma
import threading
import os
import json
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
    
    def _get_metadata_path(self):
        """Restituisce il percorso del file metadata"""
        return os.path.join(settings.VECTOR_STORE_PATH, "store_metadata.json")
    
    def _save_metadata(self, provider, document_count=None):
        """Salva i metadata del vector store"""
        metadata = {
            "provider": provider,
            "created_at": __import__('datetime').datetime.now().isoformat(),
            "document_count": document_count,
            "vector_store_type": "chroma"
        }
        try:
            with open(self._get_metadata_path(), 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"Errore nel salvare metadata: {str(e)}")
    
    def _load_metadata(self):
        """Carica i metadata del vector store"""
        try:
            with open(self._get_metadata_path(), 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def initialize(self, provider='openai', rebuild=False):
        """Inizializza lo store in modo thread-safe"""
        with self._lock:
            if self._is_initializing:
                return False
            
            if provider not in ['openai', 'gemini', 'llama']:
                raise ValueError(f"Provider non supportato: {provider}")
            
            self._is_initializing = True
            
            try:
                from .embeddings import get_embeddings
                
                metadata = self._load_metadata()
                # Chroma usa una directory, quindi verifichiamo la sua esistenza
                vector_store_exists = os.path.isdir(settings.VECTOR_STORE_PATH)
                
                if not rebuild and vector_store_exists and metadata:
                    stored_provider = metadata.get('provider')
                    
                    if stored_provider == provider:
                        print(f"Caricamento vector store Chroma esistente (provider: {stored_provider})")
                        embeddings = get_embeddings(provider)
                        
                        self._store = Chroma(
                            persist_directory=settings.VECTOR_STORE_PATH,
                            embedding_function=embeddings
                        )
                        self._provider = provider
                        
                        doc_count = self._store._collection.count() if self._store._collection else 'unknown'
                        
                        print(f"Vector store Chroma caricato! Documenti: {doc_count}")
                        return True
                    else:
                        print(f"Provider mismatch: stored={stored_provider}, requested={provider}. Sarà necessario rifare l'ingest.")
                        self._store = None
                        self._provider = provider
                        return True
                else:
                    if rebuild:
                        print("Rebuild richiesto per Chroma. La directory verrà eliminata durante l'ingest.")
                    elif not vector_store_exists:
                        print("Nessun vector store Chroma esistente trovato.")
                    else: # Manca il file metadata
                        print("Directory Chroma esistente ma metadata mancanti o corrotti.")
                    
                    self._store = None
                    self._provider = provider
                    return True
                
            except Exception as e:
                import traceback
                print(f"Errore durante l'inizializzazione del vector store Chroma: {traceback.format_exc()}")
                self._store = None
                self._provider = provider
                return False
            finally:
                self._is_initializing = False
    
    def get_store(self):
        """Restituisce lo store corrente"""
        return self._store
    
    def set_store(self, store, provider):
        """Imposta un nuovo store (dopo ingest)"""
        with self._lock:
            self._store = store
            self._provider = provider
            # Salva i metadata quando viene impostato un nuovo store
            if isinstance(store, Chroma) and provider:
                try:
                    doc_count = store._collection.count()
                    self._save_metadata(provider, doc_count)
                except Exception as e:
                    print(f"Errore durante il salvataggio dei metadati post-ingest: {e}")
                    self._save_metadata(provider, None)
    
    def is_initialized(self):
        """Verifica se lo store è stato inizializzato"""
        return self._store is not None
        
    def get_provider(self):
        """Restituisce il provider corrente"""
        return self._provider