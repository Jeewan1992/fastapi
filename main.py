from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import voyageai
import os

app = FastAPI()

# Allow Open WebUI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Voyage AI
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# Simple data models
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: str = "rerank-1"
    top_k: Optional[int] = None

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str

class RerankResponse(BaseModel):
    results: List[RerankResult]

# Check if service is running
@app.get("/")
def home():
    return {"status": "Voyage AI Reranker is running!"}

# Main reranker endpoint
@app.post("/rerank")
def rerank(request: RerankRequest):
    try:
        # Call Voyage AI
        result = vo.rerank(
            query=request.query,
            documents=request.documents,
            model=request.model,
            top_k=request.top_k
        )
        
        # Format the response
        results = []
        for item in result.results:
            results.append(RerankResult(
                index=item.index,
                relevance_score=item.relevance_score,
                document=request.documents[item.index]
            ))
        
        return RerankResponse(results=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Alternative endpoint for compatibility
@app.post("/v1/rerank")
def rerank_v1(request: RerankRequest):
    return rerank(request)
