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
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

parent_folder = "/Users/ajaynema/ai/ajay.nema.ai.agents.tmf/learning/langchain/data"

# Initialize the embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Load the CSV dataset
loader = CSVLoader(parent_folder + "/customer.csv")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
db = Chroma.from_documents(texts, embeddings, persist_directory=parent_folder)
db.persist()
print("Successfully create customer chrom db .....")

