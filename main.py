from typing import List, Optional, Dict, Any
import os
import logging
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voyage-reranker")

# ---------- Config ----------
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"

if not VOYAGE_API_KEY:
    logger.warning("⚠️ VOYAGE_API_KEY not set. Set it in Railway environment variables.")

# ---------- App ----------
app = FastAPI(
    title="Voyage Reranker API",
    description="Reranking service using Voyage AI",
    version="1.0.0"
)

# ---------- Models ----------
class Document(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None

class RerankRequest(BaseModel):
    query: str
    documents: List[Document]
    model: Optional[str] = "rerank-1"  # Default model
    top_k: Optional[int] = None

class RerankResponse(BaseModel):
    voyage_raw: Dict[str, Any]
    ranked: List[Dict[str, Any]]

# ---------- Endpoints ----------
@app.get("/")
async def root():
    """Root endpoint to verify service is running"""
    return {
        "status": "voyage-reranker is running",
        "service": "Voyage AI Reranker",
        "endpoints": {
            "health": "/health",
            "rerank": "/rerank (POST)"
        },
        "api_key_configured": bool(VOYAGE_API_KEY)
    }

@app.get("/health")
async def health():
    """Health check endpoint for Railway"""
    health_status = {
        "status": "healthy",
        "service": "voyage-reranker",
        "api_key_configured": bool(VOYAGE_API_KEY)
    }
    
    if not VOYAGE_API_KEY:
        health_status["status"] = "degraded"
        health_status["warning"] = "API key not configured"
    
    return health_status

@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    """
    Rerank documents using Voyage AI
    
    Args:
        req: RerankRequest containing query, documents, and optional parameters
        
    Returns:
        RerankResponse with raw Voyage API response and ranked documents
    """
    if not VOYAGE_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Voyage API key not configured on server. Set VOYAGE_API_KEY environment variable."
        )
    
    # Validate documents
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided for reranking")
    
    # Prepare payload for Voyage API
    payload = {
        "query": req.query,
        "documents": [d.content for d in req.documents],  # Voyage expects simple list of strings
        "model": req.model or "rerank-1"
    }
    
    if req.top_k:
        payload["top_k"] = req.top_k
    
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    logger.info(f"Reranking {len(req.documents)} documents with query: {req.query[:50]}...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(VOYAGE_RERANK_URL, json=payload, headers=headers)
    except httpx.TimeoutException:
        logger.error("Timeout contacting Voyage API")
        raise HTTPException(status_code=504, detail="Timeout contacting Voyage API")
    except httpx.RequestError as e:
        logger.exception("Error contacting Voyage API")
        raise HTTPException(status_code=502, detail=f"Error contacting Voyage API: {str(e)}")
    
    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        logger.error(f"Voyage API error (status {resp.status_code}): {err}")
        raise HTTPException(
            status_code=resp.status_code, 
            detail={"voyage_error": err, "message": "Voyage API returned an error"}
        )
    
    try:
        data = resp.json()
    except Exception as e:
        logger.error(f"Failed to parse Voyage API response: {e}")
        raise HTTPException(status_code=502, detail="Invalid response from Voyage API")
    
    # Process Voyage API response
    output = {"voyage_raw": data, "ranked": []}
    
    # Voyage API returns results in 'data' field with index and relevance_score
    if isinstance(data, dict) and "data" in data:
        results = data["data"]
        
        # Map results back to original documents
        ranked = []
        for result in results:
            doc_index = result.get("index")
            relevance_score = result.get("relevance_score", 0.0)
            
            if doc_index is not None and doc_index < len(req.documents):
                original_doc = req.documents[doc_index]
                ranked.append({
                    "id": original_doc.id,
                    "content": original_doc.content,
                    "metadata": original_doc.metadata,
                    "relevance_score": relevance_score,
                    "index": doc_index
                })
        
        output["ranked"] = ranked
        logger.info(f"Successfully reranked {len(ranked)} documents")
    else:
        # Fallback if response format is unexpected
        logger.warning("Unexpected Voyage API response format")
        output["ranked"] = [
            {
                "id": d.id, 
                "content": d.content, 
                "metadata": d.metadata
            } 
            for d in req.documents
        ]
    
    return output

# ---------- Startup Event ----------
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    port = os.getenv("PORT", "8080")
    logger.info("=" * 60)
    logger.info("🚀 Voyage Reranker API Starting")
    logger.info(f"📡 Port: {port}")
    logger.info(f"🔑 API Key Configured: {bool(VOYAGE_API_KEY)}")
    logger.info("=" * 60)

# ---------- Port Handling ----------
def get_port() -> int:
    """Get port from environment variable with fallback"""
    try:
        return int(os.getenv("PORT", "8080"))
    except ValueError:
        logger.warning("⚠️ Invalid PORT value; using 8080 as default")
        return 8080

if __name__ == "__main__":
    port = get_port()
    logger.info(f"🚀 Starting Uvicorn on 0.0.0.0:{port}")
    
    # Use the app string directly for Railway compatibility
    uvicorn.run(
        app,  # Pass app object directly instead of string
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
