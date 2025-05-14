import hashlib
import time
import threading
from functools import lru_cache

class ResponseCache:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, max_size=1000, ttl=3600):  # 1 ora TTL 
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        
    @classmethod
    def get_instance(cls, max_size=1000, ttl=3600):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_size, ttl)
        return cls._instance
        
    def _get_key(self, query, provider, model):
        # Crea una chiave unica basata su query e modello
        hash_input = f"{query}:{provider}:{model}"
        return hashlib.md5(hash_input.encode()).hexdigest()
        
    def get(self, query, provider, model):
        key = self._get_key(query, provider, model)
        cached = self.cache.get(key)
        if cached:
            timestamp, value = cached
            if time.time() - timestamp < self.ttl:
                return value
            # Scaduto
            del self.cache[key]
        return None
        
    def set(self, query, provider, model, result):
        key = self._get_key(query, provider, model)
        
        # Pulizia se raggiunta dimensione massima
        if len(self.cache) >= self.max_size:
            # Rimuove l'entry più vecchia
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]
            
        self.cache[key] = (time.time(), result)