from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
parent_folder = "/Users/ajaynema/ai/ajay.nema.ai.agents.tmf/learning/langchain/data"


# Initialize the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Now we can load the persisted database from disk, and use it as normal. 
db = Chroma(persist_directory=parent_folder, embedding_function=embeddings)
# Convert the database into a retriever
retriever = db.as_retriever()

print("DB loaded ....")

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

# Use the template to build a prompt
prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="gemma2")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
print("Chain is ready to anwser the question ....")

user_input = """
Please provide customer with name Sheryl.
"""

# Execute the pipeline
print(chain.invoke(user_input))


user_input = """
Please how many  total customers.
"""

# Execute the pipeline
print(chain.invoke(user_input))

user_input = """
Please provide total customers in Chile.
"""

# Execute the pipeline
print(chain.invoke(user_input))