from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")  # Load from .env
)
chat = ChatHuggingFace(llm=model)

messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Explain about Tsunami in 2 lines")
]

result=chat.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)