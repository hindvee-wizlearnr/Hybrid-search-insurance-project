import joblib
from pinecone import Pinecone, ServerlessSpec
from app.services.embedding_service import generate_dense_embedding

# Load TF-IDF vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Connect to Pinecone
pc = Pinecone(api_key="your-real-api-key")
index_name = "video-hybrid-search"

# Ensure index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

def query_pinecone(query: str, top_k: int = 5):
    dense_vector = generate_dense_embedding(query)

    sparse_matrix = vectorizer.transform([query]).tocoo()
    sparse_vector = {
        "indices": sparse_matrix.col.tolist(),
        "values": sparse_matrix.data.tolist()
    }

    results = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results
