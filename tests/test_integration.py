
import pytest
import os
import sys
from pprint import pprint

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from src.graph.workflow import app as graph_app
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

@pytest.mark.asyncio
async def test_end_to_end_flow():
    """
    Simulates a 'Normal Business Operation' - A full user query.
    """
    print("\n--- Starting E2E Integration Test ---")
    
    # 1. Seed the DB with a known document so the test is deterministic
    print("Seeding Qdrant...")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection_name = "agentic-engine" # MATCHES src/vectorstore.py
    
    # Ensure collection exists (main app usually creates it, but good to be safe)
    if not client.collection_exists(collection_name):
         client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )

    # 2. Run the Query
    # Query specific to the newly ingested Financial Report
    # NOW relying on Dynamic Fallback (Retrieve -> Web Search), so no manual route needed!
    inputs = {
        "question": "What was the Total Revenue and Cloud Infrastructure Cost for Q3 2025?"
    }
    print(f"Query: {inputs['question']}")
    
    final_state = await graph_app.ainvoke(inputs)
    
    # 3. Assertions
    assert "generation" in final_state, "Graph did not produce a generation!"
    answer = final_state["generation"]
    
    print("\n--- Final Answer ---")
    print(answer)
    print("--------------------")
    
    assert len(answer) > 20, "Answer was too short/empty!"
    
    # Verify we found the doc
    docs = final_state.get("documents", [])
    assert len(docs) > 0, "No relevant documents found! (Retrieval or Grading failed)"
    
    print(f"✅ E2E Test Passed. Retrieved {len(docs)} doc(s).")

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(test_end_to_end_flow())
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
