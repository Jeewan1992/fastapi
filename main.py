# main.py
from typing import List, Optional, Dict, Any
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# Port is provided by Railway in $PORT
app = FastAPI(title="Voyage Reranker API")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"

if not VOYAGE_API_KEY:
    # In production, Railway will supply this env var. Keep app usable locally (with error).
    print("Warning: VOYAGE_API_KEY not set. Set it in environment to call Voyage API.")


class Document(BaseModel):
    # Accept full objects but default to a "content" field
    id: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None


class RerankRequest(BaseModel):
    query: str
    documents: List[Document]
    model: Optional[str] = None  # e.g., "rerank-2.5" or leave None for default
    top_k: Optional[int] = None  # how many top results to return; None means return all


@app.post("/rerank")
async def rerank(req: RerankRequest):
    """
    Request body example:
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

    # Build payload expected by Voyage rerank endpoint:
    # The docs expect: { "model": "...", "query": "...", "docs": [ { "content": "..." }, ... ], ... }
    payload = {
        "query": req.query,
        "docs": [{"id": d.id, "content": d.content, "metadata": d.metadata} for d in req.documents],
    }
    if req.model:
        payload["model"] = req.model

    # Optional: send any additional instruction or argument if needed (Voyage supports extra args).
    # Call Voyage AI endpoint
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(VOYAGE_RERANK_URL, json=payload, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Error contacting Voyage API: {e}")

    if resp.status_code != 200:
        # Forward error message from Voyage if present
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise HTTPException(status_code=resp.status_code, detail={"voyage_error": err})

    data = resp.json()
    # Voyage returns ranking results — structure varies by model/version.
    # We'll extract scores and indices if present, and assemble a stable output.

    # Example expected: data might contain a list "scores" or "ranks" or "results" — include raw response.
    # But most commonly it contains a list of doc scores in the response. We'll try to be generic:
    output = {
        "voyage_raw": data,
        # Also build a convenience "ranked" list if data contains scores aligned with input docs:
        "ranked": []
    }

    # Best-effort: if Voyage returns `scores` as list of floats aligned with docs
    scores = None
    if isinstance(data, dict):
        # Check common keys
        for k in ("scores", "scores_out", "results", "ranked"):
            if k in data and isinstance(data[k], list):
                # If results list contains dicts with "score" and "index" or "id", use them:
                candidate = data[k]
                # If elements are numbers -> treat as scores
                if all(isinstance(x, (int, float)) for x in candidate):
                    scores = candidate
                    break
                # If elements are dicts and contain "score" and maybe "index" or "id"
                if all(isinstance(x, dict) for x in candidate):
                    # try to map by index or id
                    results = []
                    for item in candidate:
                        s = item.get("score")
                        idx = item.get("index")
                        did = item.get("id")
                        results.append({"id": did, "index": idx, "score": s, "raw": item})
                    output["ranked"] = results
                    # done
                    return output

    # If we found numeric scores aligned with input documents:
    if scores and len(scores) == len(req.documents):
        paired = []
        for doc, score in zip(req.documents, scores):
            paired.append({"id": doc.id, "content": doc.content, "score": score, "metadata": doc.metadata})
        # Sort by descending score (higher = better)
        paired_sorted = sorted(paired, key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)
        output["ranked"] = paired_sorted
        return output

    # Fallback: try to use "results" array containing items with "score" and "doc_index"
    # If nothing matches, return raw response and original documents
    output["ranked"] = [
        {"id": d.id, "content": d.content, "score": None, "metadata": d.metadata}
        for d in req.documents
    ]
    return output


@app.get("/")
async def root():
    return {"status": "voyage-reranker is up", "note": "POST /rerank with JSON body"}
