"""Fast HNSW->Supabase migration (no re-embedding, deduped)."""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, hnswlib
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SUPA_URL = os.getenv("SUPABASE_URL")
SUPA_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supa = create_client(SUPA_URL, SUPA_KEY)

CAPS = {"en": 100000, "ben_Beng": 30000, "hin_Deva": 50000, "tam_Taml": 50000}
hnsw_path = Path("data/hnsw/msmarco_xi.hnsw")
ids_path = Path("data/hnsw/msmarco_xi.ids.json")
sidecar = Path("data/qdrant_remote/chunks.jsonl")

ids = json.loads(ids_path.read_text(encoding="utf-8"))
index = hnswlib.Index(space="cosine", dim=384)
index.load_index(str(hnsw_path), max_elements=len(ids))
all_vecs = np.array(index.get_items(list(range(len(ids)))), dtype=np.float32)

payload_map = {}
lang_of = {}
# DEDUP sidecar by id (keep last)
with sidecar.open(encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        payload_map[j["id"]] = j
        lang_of[j["id"]] = j.get("language", "en")

# Build filtered list deduped and capped
from collections import Counter
counters = Counter()
to_migrate = []
seen = set()
for idx, cid in enumerate(ids):
    if cid in seen:
        continue
    lang = lang_of.get(cid, "en")
    if counters[lang] >= CAPS.get(lang, 50000):
        continue
    payload = payload_map.get(cid)
    if not payload:
        continue
    # dedup check
    if cid in seen:
        continue
    seen.add(cid)
    to_migrate.append((cid, all_vecs[idx], payload))
    counters[lang] += 1

print(f"Filtered {len(to_migrate)} with caps {CAPS}")
for k,v in counters.items():
    print(f"  {k}: {v}")

# Check existing count
try:
    res = supa.table("chunks").select("id", count="exact").limit(1).execute()
    existing = res.count or 0
    print(f"Supabase existing {existing}")
except Exception as e:
    print(f"Count failed {e}")
    existing = 0

# If existing is 56000, resume from there
# Build set of existing ids to skip
existing_ids = set()
if existing > 0:
    # Fetch existing ids in batches
    print("Fetching existing ids to dedup...")
    offset = 0
    batch = 1000
    while True:
        try:
            r = supa.table("chunks").select("id").range(offset, offset+batch-1).execute()
            if not r.data:
                break
            for row in r.data:
                existing_ids.add(row["id"])
            if len(r.data) < batch:
                break
            offset += batch
        except Exception as e:
            print(f"fetch existing failed {e}")
            break
    print(f"Fetched {len(existing_ids)} existing ids")

# Filter to_migrate to only new
filtered_migrate = [x for x in to_migrate if x[0] not in existing_ids]
print(f"After dedup {len(filtered_migrate)} new to insert (skipped {len(to_migrate)-len(filtered_migrate)} existing)")

BATCH = 128
t0 = time.perf_counter()
for start in range(0, len(filtered_migrate), BATCH):
    batch = filtered_migrate[start:start+BATCH]
    rows = []
    for cid, vec, payload in batch:
        rows.append({
            "id": cid,
            "source_id": payload.get("source_id"),
            "text": payload.get("text"),
            "language": payload.get("language"),
            "strategy": payload.get("strategy"),
            "embedding": vec.tolist(),
        })
    # dedup within batch (should not happen)
    # ensure unique ids in batch
    seen_batch = set()
    uniq_rows = []
    for r in rows:
        if r["id"] not in seen_batch:
            uniq_rows.append(r)
            seen_batch.add(r["id"])
    rows = uniq_rows
    for attempt in range(3):
        try:
            supa.table("chunks").upsert(rows).execute()
            break
        except Exception as e:
            msg = str(e)
            if "cannot affect row a second time" in msg:
                # fallback to insert one by one
                print(f"  batch {start} duplicate, inserting one by one")
                for row in rows:
                    try:
                        supa.table("chunks").upsert([row]).execute()
                    except Exception as e2:
                        print(f"    single {row['id']} failed {e2}")
                break
            print(f"  retry {start} attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    if (start // BATCH) % 20 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  {start+len(batch)}/{len(filtered_migrate)} {elapsed:.0f}s { (start+len(batch))/elapsed:.1f} rows/s")

print(f"Done in {time.perf_counter()-t0:.0f}s")
