FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
WORKDIR /service
# Slim runtime for Render Free (Supabase mode) — no datasets/duckdb/qdrant
COPY requirements-vercel.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY start.sh ./start.sh
RUN chmod +x ./start.sh && mkdir -p data/qdrant_remote
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["./start.sh"]
