from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
load_dotenv() # Load before importing src modules

from src.graph.workflow import app as graph_app

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agentic Reasoning Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Agentic Reasoning Engine is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "0.1.0",
        "env_check": {
            "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY")),
            "qdrant_url": os.getenv("QDRANT_URL")
        }
    }

@app.post("/invoke")
async def invoke_agent(question: str):
    """
    Invokes the agent interactions.
    """
    print(f"Received question: {question}")
    inputs = {"question": question}
    try:
        from langfuse.langchain import CallbackHandler
        langfuse_handler = CallbackHandler()
        
        # Pass the handler in the config map to graph_app.ainvoke
        result = await graph_app.ainvoke(inputs, config={"callbacks": [langfuse_handler]})
        return result
    except Exception as e:
        print(f"Error invoking graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
