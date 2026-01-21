from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from src.vectorstore import get_vectorstore, client, COLLECTION_NAME, embeddings
from qdrant_client.http import models

def ingest_data():
    """
    Ingests sample data into Qdrant.
    """
    print("---INGESTING DATA---")
    # vectorstore = get_vectorstore() # Remove early call, we do it after check
    
    docs = [
        Document(
            page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. It is built on top of LangChain and allows for cyclic flows.",
            metadata={"source": "manual"}
        ),
        Document(
            page_content="Agentic Reasoning involves systems that can plan, reflect, and correct their own mistakes. A key component is the ability to use tools and verify outputs.",
            metadata={"source": "manual"}
        ),
        Document(
            page_content="Adversarial attacks on LLMs include prompt injection, jailbreaking, and data poisoning. Defenses include input filtering and robust system prompts.",
            metadata={"source": "manual"}
        )
    ]
    
    # Ensure collection exists
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection {COLLECTION_NAME} exists.")
    except Exception:
        print(f"Creating collection {COLLECTION_NAME}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )

    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)
    print("---DATA INGESTED---")

if __name__ == "__main__":
    ingest_data()
