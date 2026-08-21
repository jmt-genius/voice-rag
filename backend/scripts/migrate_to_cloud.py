"""Migrate local Qdrant (data/qdrant_remote) to Qdrant Cloud.

Usage:
  python scripts/migrate_to_cloud.py

Reads QDRANT_HOST / QDRANT_API_KEY from .env or env vars.
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.config import settings

BATCH = 256
COLLECTION = "msmarco_xi"

cfg = settings()
cloud_host = cfg.qdrant_host
cloud_key = cfg.qdrant_api_key

if not cloud_host or not cloud_key:
    print("Set QDRANT_HOST and QDRANT_API_KEY in .env")
    sys.exit(1)

print(f"Cloud: {cloud_host}:6333")
cloud = QdrantClient(host=cloud_host, port=6333, api_key=cloud_key, https=True, timeout=60)

# Ensure collection exists on cloud with same vector config
try:
    cloud.get_collection(COLLECTION)
    print(f"Cloud collection {COLLECTION} already exists")
except Exception:
    print(f"Creating cloud collection {COLLECTION} ...")
    cloud.create_collection(COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    print("Created")

# Use HNSW + sidecar as source (avoids local Qdrant lock)
import hnswlib, numpy as np
from qdrant_client.models import PointStruct
hnsw_path = Path("data/hnsw/msmarco_xi.hnsw")
ids_path = Path("data/hnsw/msmarco_xi.ids.json")
sidecar = Path("data/qdrant_remote/chunks.jsonl")
if not hnsw_path.exists() or not ids_path.exists() or not sidecar.exists():
    print("HNSW or sidecar missing — cannot migrate without local Qdrant")
    sys.exit(1)

ids = json.loads(ids_path.read_text(encoding="utf-8"))
print(f"HNSW ids: {len(ids)}")
# Load vectors from HNSW
index = hnswlib.Index(space="cosine", dim=384)
index.load_index(str(hnsw_path), max_elements=len(ids))
all_vecs = np.array(index.get_items(list(range(len(ids)))), dtype=np.float32)
# Load payloads from sidecar
payload_map = {}
with sidecar.open(encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        payload_map[j["id"]] = {k: j[k] for k in ("source_id","text","strategy","language") if k in j}

total_local = len(ids)
print(f"Source vectors: {total_local}, payloads: {len(payload_map)}")

BATCH = 128
# Resume support: check how many already in cloud (vectors_count is None while indexing)
try:
    cloud_info = cloud.get_collection(COLLECTION)
    already = cloud_info.vectors_count or cloud_info.points_count or 0
    print(f"Cloud already has {already} vectors — resuming")
except Exception as e:
    print(f"Could not get cloud info: {e}")
    already = 0

# Upsert in batches with retry
import httpx
migrated = already
t0 = time.perf_counter()
# Skip already-migrated prefix (assumes ids are inserted in order)
start_idx = already
for start in range(start_idx, total_local, BATCH):
    batch_ids = ids[start:start+BATCH]
    batch_vecs = all_vecs[start:start+BATCH]
    points = []
    for cid, vec in zip(batch_ids, batch_vecs):
        payload = payload_map.get(cid, {})
        points.append(PointStruct(id=cid, vector=vec.tolist(), payload=payload))
    # retry loop
    for attempt in range(5):
        try:
            cloud.upsert(COLLECTION, points=points, wait=False)
            break
        except (httpx.ReadTimeout, Exception) as e:
            print(f"  retry {attempt+1}/5 for {start}: {e}")
            time.sleep(2 ** attempt)
            if attempt == 4:
                raise
    migrated += len(points)
    elapsed = time.perf_counter() - t0
    if migrated % (BATCH*10) == 0 or migrated == total_local:
        print(f"  {migrated}/{total_local} ({migrated/total_local*100:.1f}%)  {elapsed:.0f}s  {migrated/elapsed:.1f} pts/s")

print(f"Done: migrated {migrated} points in {time.perf_counter()-t0:.0f}s")
try:
    cloud_info = cloud.get_collection(COLLECTION)
    print(f"Cloud vectors: {cloud_info.vectors_count}")
except Exception as e:
    print(f"Verify failed: {e}")
