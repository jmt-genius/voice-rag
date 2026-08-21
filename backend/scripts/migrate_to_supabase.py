"""Migrate local sidecar+HNSW to Supabase pgvector (single project, Mumbai).

Uses SUPABASE_URL and SUPABASE_SERVICE_KEY from .env.
Limits per language: en (keep all), bn (30k), hi/ta (keep all) — so English is largest.
"""
import json, time, sys
from pathlib import Path
# Add both voice-rag and backend to path for flexible imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import hnswlib
from supabase import create_client
try:
    from app.config import settings
except ImportError:
    from backend.app.config import settings

cfg = settings()
SUPA_URL = cfg.supabase_url if hasattr(cfg, 'supabase_url') else None
# Fallback to env directly
import os
SUPA_URL = os.getenv("SUPABASE_URL") or getattr(cfg, "supabase_url", None)
SUPA_KEY = os.getenv("SUPABASE_SERVICE_KEY") or getattr(cfg, "supabase_service_key", None)
if not SUPA_URL or not SUPA_KEY:
    # try from .env manually
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    SUPA_URL = os.getenv("SUPABASE_URL")
    SUPA_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPA_URL or not SUPA_KEY:
    print("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in backend/.env")
    raise SystemExit(1)

supa = create_client(SUPA_URL, SUPA_KEY)

# Per-language caps — English biggest
CAPS = {"en": 45000, "ben_Beng": 30000, "hin_Deva": 50000, "tam_Taml": 50000}
# Actually keep en all (45k), cap bn at 30k

hnsw_path = Path("data/hnsw/msmarco_xi.hnsw")
ids_path = Path("data/hnsw/msmarco_xi.ids.json")
sidecar = Path("data/qdrant_remote/chunks.jsonl")

if not hnsw_path.exists() or not ids_path.exists():
    print("HNSW missing — run build_hnsw first or use Qdrant local as source")
    raise SystemExit(1)

ids = json.loads(ids_path.read_text(encoding="utf-8"))
index = hnswlib.Index(space="cosine", dim=384)
index.load_index(str(hnsw_path), max_elements=len(ids))
all_vecs = np.array(index.get_items(list(range(len(ids)))), dtype=np.float32)

# Load payloads
payload_map = {}
lang_of = {}
with sidecar.open(encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        payload_map[j["id"]] = j
        lang_of[j["id"]] = j.get("language", "en")

# Build migration list with caps
from collections import Counter, defaultdict
counters = Counter()
to_migrate = []  # list of (id, vec, payload)
for idx, cid in enumerate(ids):
    lang = lang_of.get(cid, "en")
    cap = CAPS.get(lang, 50000)
    if counters[lang] >= cap:
        continue
    payload = payload_map.get(cid)
    if not payload:
        continue
    to_migrate.append((cid, all_vecs[idx], payload))
    counters[lang] += 1

print(f"Will migrate {len(to_migrate)} chunks with caps {CAPS}")
for lang, cnt in counters.items():
    print(f"  {lang}: {cnt}")
# Clear existing (optional)
print("Clearing existing Supabase chunks...")
try:
    supa.table("chunks").delete().neq("id", "").execute()
except Exception as e:
    print(f"Clear failed (may be empty): {e}")

# Insert in batches
BATCH = 128
t0 = time.perf_counter()
for start in range(0, len(to_migrate), BATCH):
    batch = to_migrate[start:start+BATCH]
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
    # supabase upsert
    for attempt in range(3):
        try:
            supa.table("chunks").upsert(rows).execute()
            break
        except Exception as e:
            print(f"  retry batch {start} attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    if (start // BATCH) % 10 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  {start+len(batch)}/{len(to_migrate)} ({(start+len(batch))/len(to_migrate)*100:.1f}%) {elapsed:.0f}s { (start+len(batch))/elapsed:.1f} rows/s")

print(f"Done in {time.perf_counter()-t0:.0f}s")
# Verify
try:
    res = supa.table("chunks").select("language", count="exact").limit(1).execute()
    print(f"Supabase count: {res.count}")
except Exception as e:
    print(f"Count failed: {e}")
# Per-lang counts
for lang in ["en", "ben_Beng", "hin_Deva", "tam_Taml"]:
    try:
        r = supa.table("chunks").select("id", count="exact").eq("language", lang).limit(1).execute()
        print(f"  {lang}: {r.count}")
    except Exception as e:
        print(f"  {lang} count failed: {e}")
