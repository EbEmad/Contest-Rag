import redis
import json
import hashlib
from typing import Optional, List
class CacheManager:
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.client = redis.from_url(redis_url)
        self.ttl = ttl
    
    def _hash_key(self, text: str, prefix: str = "") -> str:
        """Create cache key from text."""
        return f"{prefix}:{hashlib.md5(text.encode()).hexdigest()}"
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._hash_key(text, "emb")
        cached = self.client.get(key)
        return json.loads(cached) if cached else None
    
    def set_embedding(self, text: str, embedding: List[float]):
        """Cache embedding."""
        key = self._hash_key(text, "emb")
        self.client.setex(key, self.ttl, json.dumps(embedding))
    
    def get_answer(self, query: str, project_id: str) -> Optional[str]:
        """Get cached answer."""
        key = self._hash_key(f"{project_id}:{query}", "ans")
        cached = self.client.get(key)
        return cached.decode() if cached else None
    
    def set_answer(self, query: str, project_id: str, answer: str):
        """Cache answer."""
        key = self._hash_key(f"{project_id}:{query}", "ans")
        self.client.setex(key, self.ttl, answer)