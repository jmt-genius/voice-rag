-- Run this in Supabase Dashboard → SQL Editor (paste and Run)
create extension if not exists vector;

create table if not exists chunks (
  id text primary key,
  source_id text,
  text text,
  language text,
  strategy text,
  embedding vector(384)
);

-- HNSW index for <10 ms cosine search (m=16 is Qdrant's default, here hnsw)
create index if not exists chunks_embedding_hnsw
  on chunks using hnsw (embedding vector_cosine_ops);

-- Optional: IVFFlat fallback if HNSW not available on free tier
-- create index if not exists chunks_embedding_ivfflat
--   on chunks using ivfflat (embedding vector_cosine_ops) with (lists=100);

-- For language filtering
create index if not exists chunks_language_idx on chunks (language);
