from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st

model=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
)
chat = ChatHuggingFace(llm=model)

st.header("ASK ME ANYTHING")
topic=st.selectbox("Select Topic",["Sports","Cinema","Food","Politics"])
level=st.selectbox("Select Level",["Easy","Medium","Hard"])
style=st.selectbox("Select Style",["2lines","5line","12line"])

template=PromptTemplate(
    template="give me the information about this topic {topic} , where the depth of the topic is {level} , also the no of lines are specified as {style}",
    input_variables=["topic","level","style"]
)

prompt=template.invoke({'topic':topic,"level":level,"style":style})

print(prompt)

if st.button("Shoot"):
    result=chat.invoke(prompt)
    st.write(result.content)

