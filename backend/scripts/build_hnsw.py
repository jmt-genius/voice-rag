"""Build an in-process HNSW index from chunks.jsonl for fast dense search.

Qdrant's embedded (local) mode never builds HNSW graphs, so full-scan on the
64K-chunk index blows the 50ms budget. hnswlib gives us an in-process HNSW
index whose search is ~1-3ms regardless of Qdrant's optimizer.

Recovers already-embedded vectors from leftover local Qdrant storages
(data/qdrant_quant, data/qdrant_tiny, data/qdrant_small) before embedding
the remainder.

Usage:
    python scripts/build_hnsw.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hnswlib
import numpy as np
from qdrant_client import QdrantClient
from app.config import settings
from app.retrieval import HybridRetriever

BATCH = 256
M = 16
EF_CONSTRUCTION = 100
EF_SEARCH = 64
DIM = 384

# Leftover embedded storages that may still hold real vectors.
RECOVERY_SOURCES = [
    ("data/qdrant_quant", "msmarco_xi_quant"),
    ("data/qdrant_tiny", "msmarco_xi"),
    ("data/qdrant_small", "msmarco_xi"),
]


def recover_vectors(extra_storage: str | None = None) -> dict[str, np.ndarray]:
    recovered: dict[str, np.ndarray] = {}
    sources = list(RECOVERY_SOURCES)
    if extra_storage:
        sources.append((extra_storage, "msmarco_xi"))
    for storage, collection in sources:
        path = Path(storage)
        if not (path / "collection").exists() and not (path / "collections").exists():
            continue
        try:
            client = QdrantClient(path=str(path))
            if not client.collection_exists(collection):
                client.close()
                continue
            total = 0
            offset = None
            while True:
                points, next_offset = client.scroll(
                    collection, limit=2000, offset=offset,
                    with_vectors=True, with_payload=False,
                )
                if not points:
                    break
                for p in points:
                    recovered[str(p.id)] = np.asarray(p.vector, dtype=np.float32)
                total += len(points)
                if next_offset is None:
                    break
                offset = next_offset
            client.close()
            print(f"  recovered {total} from {storage}/{collection}")
        except Exception as e:
            print(f"  skip {storage}: {type(e).__name__}: {e}")
    return recovered


def main() -> None:
    cfg = settings()
    root = Path(cfg.qdrant_path)
    sidecar = root / "chunks.jsonl"
    if not sidecar.exists():
        raise SystemExit(f"{sidecar} not found")

    print("Recovering vectors from leftover storages...")
    recovered = recover_vectors(extra_storage=str(root))
    print(f"  total recovered: {len(recovered)}")

    ids, texts = [], []
    with sidecar.open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            ids.append(item["id"])
            texts.append(item["text"])

    r = HybridRetriever(cfg)
    out = Path("data/hnsw")
    out.mkdir(parents=True, exist_ok=True)

    vectors = np.zeros((len(ids), DIM), dtype=np.float32)
    missing: list[int] = []
    for i, chunk_id in enumerate(ids):
        if chunk_id in recovered:
            vectors[i] = recovered[chunk_id]
        else:
            missing.append(i)
    print(f"Recovered vectors cover {len(ids) - len(missing)}/{len(ids)} chunks; embedding {len(missing)}...")

    t0 = time.perf_counter()
    done = 0
    for start in range(0, len(missing), BATCH):
        idx = missing[start:start + BATCH]
        embeds = list(r.embedder.embed([texts[i] for i in idx]))
        for j, vec in zip(idx, embeds):
            vectors[j] = np.asarray(vec, dtype=np.float32)
        done += len(idx)
        if done % (BATCH * 20) == 0 or done == len(missing):
            elapsed = time.perf_counter() - t0
            print(f"  {done}/{len(missing)} ({elapsed:.0f}s, {done / elapsed:.0f} chunks/s)")
    print(f"Embedding done in {time.perf_counter() - t0:.0f}s")

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    print(f"Building HNSW (M={M}, efC={EF_CONSTRUCTION})...")
    index = hnswlib.Index(space="cosine", dim=DIM)
    index.init_index(max_elements=len(vectors), ef_construction=EF_CONSTRUCTION, M=M)
    index.add_items(vectors, np.arange(len(vectors)))
    index.set_ef(EF_SEARCH)

    index_path = out / f"{cfg.collection_name}.hnsw"
    ids_path = out / f"{cfg.collection_name}.ids.json"
    index.save_index(str(index_path))
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    print(f"Saved {index_path} ({len(ids)} vectors) and {ids_path}")


if __name__ == "__main__":
    main()