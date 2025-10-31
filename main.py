from typing import List, Optional, Dict, Any
import os
import logging
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voyage-reranker")

# ---------- App ----------
app = FastAPI(title="Voyage Reranker API")

# ---------- Config ----------
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"

if not VOYAGE_API_KEY:
    logger.warning("⚠️ VOYAGE_API_KEY not set. Set it in Railway environment variables.")

# ---------- Models ----------
class Document(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None


class RerankRequest(BaseModel):
    query: str
    documents: List[Document]
    model: Optional[str] = None
    top_k: Optional[int] = None


# ---------- Endpoints ----------
@app.get("/")
async def root():
    return {"status": "voyage-reranker is up", "note": "POST /rerank with JSON body"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/rerank")
async def rerank(req: RerankRequest):
    if not VOYAGE_API_KEY:
        raise HTTPException(status_code=500, detail="Voyage API key not configured on server.")

    payload = {
        "query": req.query,
        "docs": [{"id": d.id, "content": d.content, "metadata": d.metadata} for d in req.documents],
    }
    if req.model:
        payload["model"] = req.model
    if req.top_k:
        payload["top_k"] = req.top_k

    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(VOYAGE_RERANK_URL, json=payload, headers=headers)
    except httpx.RequestError as e:
        logger.exception("Error contacting Voyage API")
        raise HTTPException(status_code=502, detail=f"Error contacting Voyage API: {e}")

    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        logger.error("Voyage API error: %s", err)
        raise HTTPException(status_code=resp.status_code, detail={"voyage_error": err})

    data = resp.json()
    output = {"voyage_raw": data, "ranked": []}

    # Try to extract scores or ranking
    if isinstance(data, dict):
        for k in ("results", "scores", "ranked"):
            if k in data and isinstance(data[k], list):
                items = data[k]
                if all(isinstance(x, dict) for x in items):
                    output["ranked"] = items
                    return output
                if all(isinstance(x, (int, float)) for x in items):
                    ranked = sorted(
                        [{"id": d.id, "content": d.content, "score": s}
                         for d, s in zip(req.documents, items)],
                        key=lambda x: x["score"],
                        reverse=True
                    )
                    output["ranked"] = ranked[: req.top_k] if req.top_k else ranked
                    return output

    # fallback if no rank info
    output["ranked"] = [{"id": d.id, "content": d.content} for d in req.documents]
    return output


# ---------- Port Handling ----------
def get_port() -> int:
    try:
        return int(os.getenv("PORT", "8080"))
    except ValueError:
        logger.warning("Invalid PORT value; using 8080 as default")
        return 8080


if __name__ == "__main__":
    port = get_port()
    logger.info(f"🚀 Starting Uvicorn on 0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
