# Extracting Structure Information using LangChain

## Project Overview
This project extracts structured information about characters from stories. The LangChain framework is used to implement the RAG (Retrieval-Augmented Generation) system. Pinecone vector database is used to store data, and Mistral AI is utilized as the LLM. The project also includes a CLI to compute embeddings and retrieve detailed character information.

---

## Features
- **Compute Embeddings:** Generate embeddings for text documents and store them in Pinecone. 
- **Retrieve Character Info:** Query detailed structured information about story characters. 
- **Preprocess Texts:** Automatically clean and prepare story files for embedding.

---

## How to Run

```bash
### 1. Clone the GitHub Repository

git clone https://github.com/pranavjadhav007/Structure-Information-using-Langchain.git

### 2. Navigate to the Project Directory

cd Extracting-Structure-Information-using-Langchain

### 3. Create a Virtual Environment

python -m venv myvenv
myvenv\Scripts\activate

### 4. Install Dependencies

pip install -r requirements.txt

### 5. Create .env file with your API keys

PINECONE_API_KEY=your_pinecone_api_key
MISTRAL_API_KEY=your_mistral_api_key

### 6. Preprocess Story Files
The text files in the story/ folder contains inverted commas not compatible with UTF-8 encoding. These cause issues when splitting using LangChain. To fix this, run the process_txt.py file:

python process_txt.py

### 7. Create the Pinecone Database
To store story data as vectors, create the Pinecone database with the desired configuration.

python database_creation.py

### 8. Run the Application

-Compute Embeddings
Generate embeddings from the story files and store them in the Pinecone Vector Database:

python main.py compute-embeddings ./story/

-Query Character Information
Retrieve structured details (in JSON format) about a character by providing the character name from the story:

python main.py get-character-info "Character Name"


