
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vectorstore import get_vectorstore
import asyncio

async def ingest():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found at {data_dir}")
        return

    vectorstore = get_vectorstore()
    
    # Loop over all markdown files
    for filename in os.listdir(data_dir):
        if not filename.endswith(".md"):
            continue
            
        file_path = os.path.join(data_dir, filename)
        print(f"--- Ingesting {file_path} ---")
        
        # 1. Load
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        print(f"Loaded {len(docs)} document(s).")
        
        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(docs)
        print(f"Split into {len(splits)} chunks.")
        
        # 3. Store
        print("Indexing to Qdrant...")
        await vectorstore.aadd_documents(splits)
    
    print("✅ Ingestion Complete!")

if __name__ == "__main__":
    asyncio.run(ingest())
