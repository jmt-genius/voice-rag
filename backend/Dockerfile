FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY start.sh ./start.sh
RUN chmod +x ./start.sh && mkdir -p data
# Optional: bake the index into the image (requires `git add -f backend/data`
# via LFS — see DEPLOY_RENDER.md Option A). Uncomment if you push data:
# COPY data ./data

ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV QDRANT_PATH=data/qdrant_remote
EXPOSE 8000

CMD ["./start.sh"]
