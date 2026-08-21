FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/start.sh ./start.sh
RUN chmod +x ./start.sh && mkdir -p data/qdrant_remote

ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV QDRANT_PATH=data/qdrant_remote
ENV OMP_NUM_THREADS=1
ENV ONNX_NUM_THREADS=1
ENV MALLOC_ARENA_MAX=2
EXPOSE 8000

CMD ["./start.sh"]
