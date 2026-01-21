import os
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize Qdrant Client
url = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(url=url, prefer_grpc=False)

# Collection Name
COLLECTION_NAME = "agentic-engine"

def get_vectorstore():
    """
    Returns the Qdrant vector store.
    """
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

def get_retriever():
    """
    Returns the retriever from the vector store.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever()
