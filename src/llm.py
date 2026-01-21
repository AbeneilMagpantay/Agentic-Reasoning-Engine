import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the Gemini Model
# Using gemini-1.5-pro as a robust default for reasoning tasks
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,
)
