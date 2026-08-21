"""Migrate sidecar directly to Supabase via embedding (no HNSW needed).

Uses caps: en keep all (45k), bn 30k, hi/ta keep all.
"""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter
from supabase import create_client
from app.config import settings
from fastembed import TextEmbedding
import os
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SUPA_URL = os.getenv("SUPABASE_URL")
SUPA_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPA_URL or not SUPA_KEY:
    print("Missing SUPABASE env")
    sys.exit(1)
supa = create_client(SUPA_URL, SUPA_KEY)
cfg = settings()
# Caps
CAPS = {"en": 100000, "ben_Beng": 30000, "hin_Deva": 50000, "tam_Taml": 50000}
sidecar = Path("data/qdrant_remote/chunks.jsonl")
# Read all chunks
all_chunks = []
with sidecar.open(encoding="utf-8") as f:
    for line in f:
        j = json.loads(line)
        all_chunks.append(j)
print(f"Sidecar total {len(all_chunks)}")
# Apply caps per language
counters = Counter()
filtered = []
for ch in all_chunks:
    lang = ch.get("language", "en")
    cap = CAPS.get(lang, 50000)
    if counters[lang] >= cap:
        continue
    filtered.append(ch)
    counters[lang] += 1
print(f"Filtered to {len(filtered)} with caps {CAPS}")
for k,v in counters.items():
    print(f"  {k}: {v}")
# Clear existing
print("Clearing Supabase...")
try:
    supa.table("chunks").delete().neq("id", "").execute()
    print("Cleared")
except Exception as e:
    print(f"Clear failed: {e}")
# Embed and insert in batches
embedder = TextEmbedding(model_name=cfg.embedding_model)
BATCH = 64
t0 = time.perf_counter()
for start in range(0, len(filtered), BATCH):
    batch = filtered[start:start+BATCH]
    texts = [c["text"] for c in batch]
    vecs = list(embedder.embed(texts))
    rows = []
    for ch, vec in zip(batch, vecs):
        rows.append({
            "id": ch["id"],
            "source_id": ch.get("source_id"),
            "text": ch["text"],
            "language": ch.get("language"),
            "strategy": ch.get("strategy"),
            "embedding": vec.tolist(),
        })
    # upsert
    for attempt in range(3):
        try:
            supa.table("chunks").upsert(rows).execute()
            break
        except Exception as e:
            print(f"  retry {start} {e}")
            time.sleep(2**attempt)
    if (start // BATCH) % 20 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  {start+len(batch)}/{len(filtered)} {elapsed:.0f}s { (start+len(batch))/elapsed:.1f} rows/s")
print(f"Done {len(filtered)} in {time.perf_counter()-t0:.0f}s")
