from langchain_core.prompts import ChatPromptTemplate

chat_template=ChatPromptTemplate([
    ('system',"you are specified in this {domain}"),
    ("human","tell me about this {topic}")
])

prompt=chat_template.invoke({"domain":"cricket","topic":"duckout"})
print(prompt)