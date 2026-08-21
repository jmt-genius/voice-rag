from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .chunking import Chunk
from .config import Settings
from .contracts import Citation

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def terms(text: str) -> set[str]:
    return {x.lower() for x in TOKEN_RE.findall(text) if len(x) > 2}


@dataclass
class FastHit:
    id: int | str
    score: float


class FastSearchClient:
    """In-process HNSW dense search (hnswlib) with Qdrant API compatibility.

    Qdrant's embedded/local mode never builds HNSW graphs, so a 64K-chunk
    collection is full-scanned on every query (~115ms) and blows the 50ms
    budget. Loading the same vectors into hnswlib brings dense search down
    to ~1-3ms. All other client methods delegate to the wrapped QdrantClient.
    """

    def __init__(self, base: QdrantClient, index, ids: list[str]):
        self._base = base
        self._index = index
        self._ids = ids

    def search(self, collection_name: str, query_vector: list[float] | np.ndarray, limit: int = 10, **kwargs):
        query = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query /= norm
        labels, scores = self._index.knn_query(query.reshape(1, -1), k=limit)
        return [FastHit(id=self._ids[i], score=float(score)) for i, score in zip(labels[0], scores[0])]

    def __getattr__(self, name: str):
        return getattr(self._base, name)


class HybridRetriever:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.root = Path(cfg.qdrant_path)
        if cfg.qdrant_host:
            self.client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
        else:
            self.client = QdrantClient(path=str(self.root))
        self.embedder = TextEmbedding(model_name=cfg.embedding_model)
        self.lexical: dict[str, list[str]] = {}
        self.chunk_meta: dict[str, dict] = {}
        self.idf: dict[str, float] = {}
        self._load_sidecar()
        self._load_fast_index()

    def _load_fast_index(self) -> None:
        index_path = Path("data/hnsw") / f"{self.cfg.collection_name}.hnsw"
        ids_path = Path("data/hnsw") / f"{self.cfg.collection_name}.ids.json"
        if not (index_path.exists() and ids_path.exists()):
            return
        try:
            import hnswlib
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            index = hnswlib.Index(space="cosine", dim=384)
            index.load_index(str(index_path), max_elements=max(len(ids), 1))
            self.client = FastSearchClient(self.client, index, ids)
        except Exception:
            pass

    def _load_sidecar(self) -> None:
        path = self.root / "chunks.jsonl"
        if not path.exists():
            return
        index: dict[str, list[str]] = defaultdict(list)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                self.chunk_meta[item["id"]] = item
                for term in terms(item["text"]):
                    index[term].append(item["id"])
        self.lexical = dict(index)
        total = len(self.chunk_meta)
        self.idf = {term: math.log((total + 1) / (len(ids) + 1)) + 1.0 for term, ids in self.lexical.items()}
        self.idf_max = math.log(total + 1) + 1.0

    def create_collection(self) -> None:
        if self.client.collection_exists(self.cfg.collection_name):
            return
        # paraphrase-multilingual-MiniLM-L12-v2 output dimensionality.
        self.client.create_collection(self.cfg.collection_name, vectors_config=VectorParams(size=384, distance=Distance.COSINE))

    def index(self, chunks: Iterable[Chunk], batch_size: int = 256) -> int:
        self.create_collection()
        total = 0
        batch: list[Chunk] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) == batch_size:
                total += self._upsert(batch)
                batch = []
        if batch:
            total += self._upsert(batch)
        return total

    def _upsert(self, batch: list[Chunk]) -> int:
        vectors = list(self.embedder.embed([c.text for c in batch]))
        points = [PointStruct(id=c.id, vector=v.tolist(), payload={
            "source_id": c.source_id, "text": c.text, "strategy": c.strategy,
            "language": c.language,
        }) for c, v in zip(batch, vectors)]
        self.client.upsert(self.cfg.collection_name, points=points, wait=True)
        return len(points)

    def search(self, question: str, limit: int | None = None, language: str | None = None) -> list[Citation]:
        if not self.chunk_meta:
            return []
        limit = limit or self.cfg.top_k
        vector = next(self.embedder.query_embed([question]))
        # A language-scoped search must see a larger dense pool: the global
        # top-k can be dominated by other languages, so filter after a wider
        # search instead of shrinking recall to the language's global rank.
        dense = self.client.search(self.cfg.collection_name, query_vector=vector.tolist(), limit=(limit * 25 if language else limit * 3))
        # Lexical candidates make named entities and exact spoken phrases recoverable.
        lexical_scores: dict[str, int] = defaultdict(int)
        for term in terms(question):
            for chunk_id in self.lexical.get(term, ()):
                if language and self.chunk_meta.get(chunk_id, {}).get("language") != language:
                    continue
                lexical_scores[chunk_id] += 1
        dense_rank = {}
        for rank, hit in enumerate(dense, 1):
            meta = self.chunk_meta.get(str(hit.id))
            if language and (not meta or meta.get("language") != language):
                continue
            dense_rank[str(hit.id)] = rank
        lexical_rank = {cid: rank for rank, (cid, _) in enumerate(sorted(lexical_scores.items(), key=lambda x: -x[1]), 1)}
        candidates = set(dense_rank) | set(list(lexical_rank)[:limit * 8])
        # Reciprocal Rank Fusion, then remove siblings from the same source when possible.
        fused = sorted(candidates, key=lambda cid: 1 / (60 + dense_rank.get(cid, 10_000)) + 1 / (60 + lexical_rank.get(cid, 10_000)), reverse=True)
        out, sources = [], set()
        for cid in fused:
            meta = self.chunk_meta.get(cid)
            if not meta or (meta["source_id"] in sources and len(out) >= 2):
                continue
            dense_score = next((float(hit.score) for hit in dense if str(hit.id) == cid), 0.0)
            out.append(Citation(source_id=meta["source_id"], text=meta["text"], score=round(dense_score, 4), strategy=meta["strategy"]))
            sources.add(meta["source_id"])
            if len(out) == limit:
                break
        return out
