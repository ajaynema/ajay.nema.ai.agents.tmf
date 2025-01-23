from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv()

chat_history = []
chat_history.append(SystemMessage("you are a math expert"))

llm = ChatOllama(model="gemma2")
while True :
    query = input("You : ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(result.content)
