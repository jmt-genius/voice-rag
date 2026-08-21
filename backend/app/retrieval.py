from __future__ import annotations

import json
import math
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Low-latency tuning: limit thread pools in containers to prevent memory spikes
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "1"))
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("ONNX_NUM_THREADS", os.getenv("ONNX_NUM_THREADS", "1"))
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

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
        # Supabase pgvector (free, Mumbai) — preferred when configured
        self.supabase = None
        if getattr(cfg, "supabase_url", None) and getattr(cfg, "supabase_service_key", None):
            try:
                from supabase import create_client as _create
                self.supabase = _create(cfg.supabase_url, cfg.supabase_service_key)
            except Exception:
                self.supabase = None
        if cfg.qdrant_host:
            kwargs = {"host": cfg.qdrant_host, "port": cfg.qdrant_port}
            if cfg.qdrant_api_key:
                kwargs["api_key"] = cfg.qdrant_api_key
                kwargs["https"] = True
            self.client = QdrantClient(**kwargs)
        else:
            try:
                self.client = QdrantClient(path=str(self.root))
            except Exception:
                self.client = None
        # If Supabase is configured, we don't need local HNSW/Qdrant for dense
        if self.supabase is not None:
            self.client = None
        threads_count = int(os.getenv("EMBEDDING_THREADS", "1"))
        self.embedder = TextEmbedding(model_name=cfg.embedding_model, threads=threads_count)
        import gc; gc.collect()
        self.lexical: dict[str, list[str]] = {}
        self.lexical_per_lang: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.chunk_meta: dict[str, dict] = {}
        self.idf: dict[str, float] = {}
        self.idf_max: float = 1.0
        self._query_vec_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_vec_cache_max = 512
        self._per_lang_index: dict[str, object] = {}
        self._per_lang_ids: dict[str, list[str]] = {}
        # In Supabase mode: skip loading 94k rows into RAM — just set sentinel
        # so search() knows to proceed. chunk_meta is populated on-demand per result.
        if self.supabase is not None:
            self.chunk_meta["__supabase__"] = {}
        else:
            self._load_sidecar()
            self._load_fast_index()

    def _load_fast_index(self) -> None:
        # Cloud/Supabase mode: keep container under 512 MB — skip 500 MB HNSW load
        if self.supabase is not None or self.cfg.qdrant_host:
            return
        index_dir = Path("data/hnsw")
        index_path = index_dir / f"{self.cfg.collection_name}.hnsw"
        ids_path = index_dir / f"{self.cfg.collection_name}.ids.json"
        if not (index_path.exists() and ids_path.exists()):
            return
        try:
            import hnswlib
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            index = hnswlib.Index(space="cosine", dim=384)
            index.load_index(str(index_path), max_elements=max(len(ids), 1))
            # LowLatency.pdf HNSW tuning: ef_search 40-80 yields >95% recall at 5-12ms.
            # Build used M=16 efC=128; enforce ef=64 (mid-range) after load.
            index.set_ef(64)
            self.client = FastSearchClient(self.client, index, ids)
            # Per-language HNSW shards for true language-scoped search
            # (LowLatency2.pdf sec 7 & 10): each language gets its own
            # HNSW graph so dense search is O(log n) not O(n).
            # First try loading pre-built per-language shards directly from disk:
            loaded_all_per_lang = True
            for lang in ["tam_Taml", "hin_Deva", "en", "ben_Beng"]:
                lang_hnsw = index_dir / f"{self.cfg.collection_name}_{lang}.hnsw"
                lang_ids_file = index_dir / f"{self.cfg.collection_name}_{lang}.ids.json"
                if lang_hnsw.exists() and lang_ids_file.exists():
                    lang_ids = json.loads(lang_ids_file.read_text(encoding="utf-8"))
                    lang_idx = hnswlib.Index(space="cosine", dim=384)
                    lang_idx.load_index(str(lang_hnsw), max_elements=len(lang_ids))
                    lang_idx.set_ef(64)
                    self._per_lang_index[lang] = lang_idx
                    self._per_lang_ids[lang] = lang_ids
                else:
                    loaded_all_per_lang = False
            if not loaded_all_per_lang and self._per_lang_index:
                pass
            elif not loaded_all_per_lang:
                all_vecs = np.array(index.get_items(list(range(len(ids)))), dtype=np.float32)
                lang_to_pos: dict[str, list[int]] = defaultdict(list)
                for pos, cid in enumerate(ids):
                    lang = self.chunk_meta.get(cid, {}).get("language") or "en"
                    lang_to_pos[lang].append(pos)
                for lang, pos_list in lang_to_pos.items():
                    lang_vecs = all_vecs[pos_list]
                    lang_ids = [ids[p] for p in pos_list]
                    n = len(lang_ids)
                    lang_idx = hnswlib.Index(space="cosine", dim=384)
                    lang_idx.init_index(max_elements=n, M=16, ef_construction=128)
                    lang_idx.add_items(lang_vecs, list(range(n)))
                    lang_idx.set_ef(64)
                    self._per_lang_index[lang] = lang_idx
                    self._per_lang_ids[lang] = lang_ids
                    # Save for subsequent instant startups
                    try:
                        lang_idx.save_index(str(index_dir / f"{self.cfg.collection_name}_{lang}.hnsw"))
                        (index_dir / f"{self.cfg.collection_name}_{lang}.ids.json").write_text(json.dumps(lang_ids), encoding="utf-8")
                    except Exception:
                        pass
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
                lang = item.get("language") or "en"
                for term in terms(item["text"]):
                    index[term].append(item["id"])
                    self.lexical_per_lang[lang][term].append(item["id"])
        self.lexical = dict(index)
        # freeze per-lang shards
        self.lexical_per_lang = {k: dict(v) for k, v in self.lexical_per_lang.items()}
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

    def _get_query_vector(self, question: str) -> np.ndarray:
        # Tier-1 vector cache (LowLatency2.pdf sec 4): repeated chip prompts hit in <1ms.
        cached = self._query_vec_cache.get(question)
        if cached is not None:
            self._query_vec_cache.move_to_end(question)
            return cached
        vec = next(self.embedder.query_embed([question]))
        arr = np.asarray(vec, dtype=np.float32)
        self._query_vec_cache[question] = arr
        if len(self._query_vec_cache) > self._query_vec_cache_max:
            self._query_vec_cache.popitem(last=False)
        return arr

    def search(self, question: str, limit: int | None = None, language: str | None = None) -> list[Citation]:
        if not self.chunk_meta:
            return []
        limit = limit or self.cfg.top_k

        # In Supabase mode: fetch dense results from pgvector, then do lightweight
        # lexical re-ranking only on those rows (no 94k-row prefetch).
        if self.supabase is not None:
            return self._search_supabase(question, limit, language)

        # Local HNSW / Qdrant path — lexical index is pre-built from sidecar.
        lexical_scores: dict[str, int] = defaultdict(int)

        def _lexical():
            shard = self.lexical_per_lang.get(language, self.lexical) if language else self.lexical
            for term in terms(question):
                idf = self.idf.get(term, self.idf_max)
                if idf < 2.0:
                    continue
                for chunk_id in shard.get(term, ()):
                    lexical_scores[chunk_id] += 1

        import threading
        lex_thread = threading.Thread(target=_lexical, daemon=True)
        lex_thread.start()
        vector = self._get_query_vector(question)
        lex_thread.join()

        dense_k = limit * 3
        if language and language in self._per_lang_index:
            query = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
            lang_idx = self._per_lang_index[language]
            lang_ids = self._per_lang_ids[language]
            k = min(dense_k, len(lang_ids))
            labels, scores = lang_idx.knn_query(query.reshape(1, -1), k=k)
            dense = [FastHit(id=lang_ids[i], score=float(s)) for i, s in zip(labels[0], scores[0])]
        else:
            dense = self.client.search(self.cfg.collection_name, query_vector=vector.tolist(), limit=dense_k) if self.client else []

        dense_rank = {}
        dense_scores: dict[str, float] = {}
        for rank, hit in enumerate(dense, 1):
            hid = str(hit.id)
            meta = self.chunk_meta.get(hid)
            if language and (not meta or meta.get("language") != language):
                continue
            dense_rank[hid] = rank
            dense_scores[hid] = float(hit.score)
        lexical_rank = {cid: rank for rank, (cid, _) in enumerate(sorted(lexical_scores.items(), key=lambda x: -x[1]), 1)}
        candidates = set(dense_rank) | set(list(lexical_rank)[:limit * 8])
        fused = sorted(candidates, key=lambda cid: 1 / (60 + dense_rank.get(cid, 10_000)) + 1 / (60 + lexical_rank.get(cid, 10_000)), reverse=True)
        out, sources = [], set()
        for cid in fused:
            meta = self.chunk_meta.get(cid)
            if not meta or (meta["source_id"] in sources and len(out) >= 2):
                continue
            out.append(Citation(source_id=meta["source_id"], text=meta["text"], score=round(dense_scores.get(cid, 0.0), 4), strategy=meta["strategy"]))
            sources.add(meta["source_id"])
            if len(out) == limit:
                break
        return out

    def _search_supabase(self, question: str, limit: int, language: str | None) -> list[Citation]:
        """Supabase mode: dense via pgvector match_chunks RPC + lightweight in-process
        lexical re-ranking on the small result set. No 94k-row prefetch needed."""
        vector = self._get_query_vector(question)
        dense_k = limit * 3
        try:
            res = self.supabase.rpc("match_chunks", {
                "query_embedding": vector.tolist(),
                "match_language": language,
                "match_count": dense_k
            }).execute()
            rows = res.data or []
        except Exception:
            rows = []
        if not rows:
            return []

        # Lightweight lexical scoring over just the returned rows (not 94k rows)
        query_terms = terms(question)
        dense_rank: dict[str, int] = {}
        dense_scores: dict[str, float] = {}
        lex_scores: dict[str, int] = defaultdict(int)
        row_meta: dict[str, dict] = {}

        for rank, row in enumerate(rows, 1):
            rid = str(row["id"])
            dense_rank[rid] = rank
            dense_scores[rid] = float(row.get("score", 0))
            row_meta[rid] = row
            # Score lexical overlap on just this row's text
            chunk_terms = terms(row.get("text", ""))
            lex_scores[rid] = len(query_terms & chunk_terms)

        lexical_rank = {cid: rank for rank, (cid, _) in enumerate(sorted(lex_scores.items(), key=lambda x: -x[1]), 1)}
        candidates = list(dense_rank.keys())
        fused = sorted(
            candidates,
            key=lambda cid: 1 / (60 + dense_rank.get(cid, 10_000)) + 1 / (60 + lexical_rank.get(cid, 10_000)),
            reverse=True,
        )
        out, sources = [], set()
        for cid in fused:
            row = row_meta.get(cid)
            if not row:
                continue
            sid = row.get("source_id", cid)
            if sid in sources and len(out) >= 2:
                continue
            out.append(Citation(
                source_id=sid,
                text=row.get("text", ""),
                score=round(dense_scores.get(cid, 0.0), 4),
                strategy=row.get("strategy", ""),
            ))
            sources.add(sid)
            if len(out) == limit:
                break
        return out
