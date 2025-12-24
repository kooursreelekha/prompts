from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")  # Load from .env
)

chat = ChatHuggingFace(llm=llm)
st.header("Research tool")
user_input=st.text_input("Enter prompt")

if st.button("Enter"):
    response = chat.invoke(user_input)
    st.write(response.content) 