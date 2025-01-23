from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

import os

load_dotenv()

template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

llm = ChatOllama(model="gemma2")
chain = template|llm

result = chain.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?"
    }
)

print(result.content)