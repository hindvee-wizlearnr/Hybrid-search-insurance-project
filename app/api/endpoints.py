from fastapi import APIRouter, Query
from app.services.pinecone_service import query_pinecone

router = APIRouter()

@router.get("/search")
async def hybrid_search(query: str = Query(..., description="Search query")):
    results = query_pinecone(query)
    return [
        {
            "score": match.score,
            "text": match.metadata.get("text", "N/A")
        }
        for match in results.matches
    ]
