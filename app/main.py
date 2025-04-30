from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title="Hybrid Search API", description="Semantic + keyword hybrid search")

app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Hybrid Search API is running"}
