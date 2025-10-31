# Voyage Reranker FastAPI

Simple FastAPI wrapper around Voyage AI's reranker endpoint.

## Run locally

Set env var:
```bash
export VOYAGE_API_KEY="sk-..."
export PORT=8000
uvicorn main:app --host 0.0.0.0 --port $PORT
