# main.py
from typing import List, Optional, Dict, Any
import os
import logging
import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import uvicorn

# ----- Logging -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voyage-reranker")

# ----- App -----
app = FastAPI(title="Voyage Reranker API")

# ----- Config -----
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"

if not VOYAGE_API_KEY:
    logger.warning("VOYAGE_API_KEY not set. Set it in environment to call Voyage API.")

# ----- Models -----
class Document(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None


class RerankRequest(BaseModel):
    query: str
    documents: List[Document]
    model: Optional[str] = None  # e.g., "rerank-2.5"
    top_k: Optional[int] = None  # how many top results to return; None -> return all


# ----- Endpoints -----
@app.get("/")
async def root():
    """Health / root endpoint."""
    return {"status": "voyage-reranker is up", "note": "POST /rerank with JSON body"}


@app.post("/rerank")
async def rerank(req: RerankRequest):
    """
    POST /rerank
    Body example:
    {
      "query": "What is the return policy?",
      "documents": [
        {"id": "d1", "content": "Our returns last 30 days ..."},
        {"id": "d2", "content": "We do not accept returns for sale items ..."}
      ],
      "model": "rerank-2.5",
      "top_k": 5
    }
    """
    if not VOYAGE_API_KEY:
        raise HTTPException(status_code=500, detail="Voyage API key not configured on server.")

    # Build payload for Voyage rerank endpoint
    payload: Dict[str, Any] = {
        "query": req.query,
        "docs": [{"id": d.id, "content": d.content, "metadata": d.metadata} for d in req.documents],
    }
    if req.model:
        payload["model"] = req.model
    if req.top_k is not None:
        payload["top_k"] = req.top_k

    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(VOYAGE_RERANK_URL, json=payload, headers=headers)
    except httpx.RequestError as e:
        logger.exception("Error contacting Voyage API")
        raise HTTPException(status_code=502, detail=f"Error contacting Voyage API: {e}")

    # Forward non-200 responses as 502/400 depending on status
    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        logger.error("Voyage API returned non-200: %s %s", resp.status_code, err)
        raise HTTPException(status_code=resp.status_code, detail={"voyage_error": err})

    try:
        data = resp.json()
    except Exception:
        logger.exception("Failed to parse Voyage response as JSON")
        raise HTTPException(status_code=502, detail="Invalid JSON from Voyage API")

    # Prepare convenient output
    output: Dict[str, Any] = {"voyage_raw": data, "ranked": []}

    # Heuristic parsing:
    # 1) Check common keys that may contain rankings/scores.
    scores = None
    if isinstance(data, dict):
        for k in ("scores", "scores_out", "results", "ranked"):
            if k in data and isinstance(data[k], list):
                candidate = data[k]
                # if list of numbers -> numeric scores aligned with input docs
                if all(isinstance(x, (int, float)) for x in candidate):
                    scores = candidate
                    break
                # if list of dicts that include score/id/index -> map those
                if all(isinstance(x, dict) for x in candidate):
                    results = []
                    for item in candidate:
                        s = item.get("score")
                        idx = item.get("index")
                        did = item.get("id")
                        results.append({"id": did, "index": idx, "score": s, "raw": item})
                    output["ranked"] = results
                    return output

    # 2) If numeric scores found and length matches input docs, pair & sort
    if scores and len(scores) == len(req.documents):
        paired = []
        for doc, score in zip(req.documents, scores):
            paired.append({"id": doc.id, "content": doc.content, "score": score, "metadata": doc.metadata})
        paired_sorted = sorted(paired, key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)
        if req.top_k:
            paired_sorted = paired_sorted[: req.top_k]
        output["ranked"] = paired_sorted
        return output

    # 3) If data contains a 'ranked' list with doc references, try to use it
    if isinstance(data, dict) and "ranked" in data and isinstance(data["ranked"], list):
        output["ranked"] = data["ranked"][: req.top_k] if req.top_k else data["ranked"]
        return output

    # 4) Fallback: return original docs (unscored) and raw response
    output["ranked"] = [
        {"id": d.id, "content": d.content, "score": None, "metadata": d.metadata} for d in req.documents
    ]
    return output


# ----- Run server reading Railway PORT env var -----
def _get_port() -> int:
    raw = os.environ.get("PORT", "")
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid PORT value '%s', falling back to 8000", raw)
    return 8000


if __name__ == "__main__":
    port = _get_port()
    logger.info("Starting Uvicorn on 0.0.0.0:%d", port)
    # Use uvicorn.run from Python so env-var port is respected on Railway
    uvicorn.run("main:app", host="0.0.0.0", port=port)
