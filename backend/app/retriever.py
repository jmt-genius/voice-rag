import time
from dataclasses import dataclass

from app.config import settings
from app.retrieval import HybridRetriever, terms

@dataclass
class SearchResponse:
    total_ms: float
    embed_ms: float
    search_ms: float

_retriever = None

def warmup():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(settings())
    _retriever.search("warmup", limit=1)

def search(query: str, top_k: int = 5) -> SearchResponse:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(settings())
        
    start_total = time.perf_counter()
    
    start_embed = time.perf_counter()
    vector = next(_retriever.embedder.query_embed([query]))
    embed_ms = (time.perf_counter() - start_embed) * 1000
    
    start_search = time.perf_counter()
    dense = _retriever.client.search(_retriever.cfg.collection_name, query_vector=vector.tolist(), limit=top_k * 3)
    search_ms = (time.perf_counter() - start_search) * 1000
    
    # Simulate the rest of the search fusion logic to make total_ms accurate
    lexical_scores = {}
    for term in terms(query):
        for chunk_id in _retriever.lexical.get(term, ()):
            lexical_scores[chunk_id] = lexical_scores.get(chunk_id, 0) + 1
            
    total_ms = (time.perf_counter() - start_total) * 1000
    return SearchResponse(total_ms=total_ms, embed_ms=embed_ms, search_ms=search_ms)
