import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize LLM backend
# Using gemini-1.5-pro as the primary inference engine
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,
)
