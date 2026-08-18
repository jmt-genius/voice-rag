"""Stream MSMARCO-XI and build the multi-resolution Qdrant index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb
from app.chunking import chunk_passage
from app.config import settings
from app.retrieval import HybridRetriever


def pick(record: dict, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


LANGUAGES = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur")
LANGUAGE_TO_FILE = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal",
    "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "ta": "tam",
    "te": "tel", "ur": "urd",
}


def records(languages: list[str], split: str, limit: int):
    """Read one language Parquet at a time through DuckDB's range-capable HTTP client.

    The Hugging Face `datasets` streaming client follows the source's large Parquet redirect
    into memory on Windows. DuckDB uses HTTP range reads instead. Validation is the safe
    default (about 460 MB/language); train files can be several GB and need more RAM.
    """
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs; LOAD httpfs")
    try:
        for language in languages:
            stem = LANGUAGE_TO_FILE.get(language)
            if not stem:
                raise ValueError(f"Unsupported language code: {language}")
            url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/{split}/{stem}{'val' if split == 'validation' else 'train'}.parquet"
            cursor = connection.execute("SELECT query_id, target_lang, passages FROM read_parquet(?) LIMIT ?", [url, limit])
            while rows := cursor.fetchmany(128):
                for query_id, target_lang, passages in rows:
                    yield {"query_id": query_id, "target_lang": target_lang, "passages": passages}
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument("--languages", nargs="*", default=["hi"])
    parser.add_argument("--split", choices=("validation", "train"), default="validation")
    args = parser.parse_args()
    cfg, retriever = settings(), HybridRetriever(settings())
    sidecar = Path(cfg.qdrant_path) / "chunks.jsonl"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if sidecar.exists():
        raise SystemExit(f"{sidecar} already exists; use a new QDRANT_PATH to avoid duplicate indexing.")
    batch, documents, chunks = [], 0, 0
    with sidecar.open("w", encoding="utf-8") as out:
        for row in records(args.languages, args.split, args.limit):
            language = str(row.get("target_lang", "unknown"))
            passage_set = row.get("passages", {})
            passages = passage_set.get("Translated_passages", []) if isinstance(passage_set, dict) else []
            query_id = row.get("query_id", documents)
            for position, text in enumerate(passages):
                if not isinstance(text, str) or not text.strip():
                    continue
                child_chunks = chunk_passage(text, f"{language}:{query_id}:{position}", language)
                for chunk in child_chunks:
                    out.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")
                batch.extend(child_chunks)
            documents += 1
            if len(batch) >= 512:
                chunks += retriever.index(batch)
                batch.clear()
                print(f"indexed documents={documents} chunks={chunks}")
            if documents >= args.limit:
                break
        if batch:
            chunks += retriever.index(batch)
    print(json.dumps({"documents": documents, "chunks": chunks, "path": str(sidecar)}))


if __name__ == "__main__":
    main()
