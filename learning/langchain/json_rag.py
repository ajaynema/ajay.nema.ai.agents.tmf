import sys
import ollama
import chromadb
import requests
from datetime import datetime

# Fetch data from Culver's API
response = requests.get('https://culvers-fotd.joe.workers.dev/')
data = response.json()

# Get the current date
today = datetime.now().strftime("%B %d")

# Add the current date as the first document
documents = [f"Today's date is {today}."]

# Process the Culver's data
for item in data:
    location = item['Location']
    address = item['Address']
    flavor = item['Flavor']
    document = f"The flavors of the day for today at the {location} Culver's location at {address} is {flavor}."
    documents.append(document)

# Initialize the ChromaDB client and create a collection
client = chromadb.Client()
collection = client.create_collection(name="docs")

# Store each document in a vector embedding database
for i, d in enumerate(documents):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )

# Check if a prompt is provided as an argument
if len(sys.argv) > 1:
    prompt = ' '.join(sys.argv[1:])
else:
    prompt = "What is the flavor of the day at Culver's?"

# Generate an embedding for the prompt and retrieve the most relevant document
response = ollama.embeddings(
    prompt=prompt,
    model="mxbai-embed-large"
)
results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)
data = results['documents'][0][0]

# Generate a response combining the prompt and data retrieved in the previous step
output = ollama.generate(
    model="mxbai-embed-large",
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])