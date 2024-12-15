import os
from dotenv import load_dotenv
load_dotenv(".env")
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ["MISTRAL_API_KEY"] = os.getenv('assign_key')

from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
import time
import argparse

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model="mistral-large-latest")


def compute_embeddings(stories_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    loader = DirectoryLoader(stories_path, glob="**/*.txt",loader_cls=TextLoader)
    docs = loader.load()
    texts = text_splitter.split_documents(docs)

    embedding_model = PineconeEmbeddings(model="multilingual-e5-large")

    vectore_store = PineconeVectorStore.from_documents(
        documents=texts,
        index_name="asign_db",
        embedding=embedding_model,
        namespace="wondervector5000"
    )
    print("Embeddings computed and stored successfully.")

def get_character_info(character_name):
    embedding_model = PineconeEmbeddings(model="multilingual-e5-large")

    vectore_store = PineconeVectorStore(
        index_name="asign_db",
        embedding=embedding_model,
        namespace="wondervector5000"
    )
    retriever = vectore_store.as_retriever(search_kwargs={"k": 6})
    template = """
    You are good in retrieving detailed information about a character in a story from the {context} provided. 
    Given the character name, search through the processed embeddings and return the relevant details in a structured JSON format. 
    If the character name is not found, then return JSON object with name and data not found.
    The JSON response should contain the following keys:

    - name: The name of the character.
    - storyTitle: The title of the story where the character appears.
    - summary: A brief summary of the character's story.
    - relations: A list of relationships the character has with other characters in the story, with each relation including the character's name and their relationship to the main character.
    - characterType: The role of the character (e.g., protagonist, villain, side character).

    Input:
    - Character Name: {input}

    Output:
        "name": "{input}",
        "storyTitle": "storyTitle",
        "summary": "summary",
        "relations": [
            "name": "relatedCharacter1Name", "relation": "relation1" ,
            "name": "relatedCharacter2Name", "relation": "relation2" 
        ],
        "characterType": "characterType"
    """

    prompt_for_character_info = PromptTemplate(
        template=template,
        input_variables=["context", "input"]
    )

    expected_json = {
        "name": "string",
        "storyTitle": "string",
        "summary": "string",
        "relations": [{"name": "string", "relation": "string"}],
        "characterType": "string"
    }
    json_parser = JsonOutputParser(json_schema=expected_json)

    prompt = PromptTemplate(template=template, input_variables=["context", "input"])

    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt_for_character_info
        | model
        | json_parser
    )

    response = rag_chain.invoke("Miss Devlin")

    print(response)


def main():
    parser = argparse.ArgumentParser(description="CLI for application")
    subparsers = parser.add_subparsers(dest="command")

    parser_compute = subparsers.add_parser("compute-embeddings", help="Compute embeddings for stories")
    parser_compute.add_argument("stories_path", type=str, help="Path to the folder containing story files")

    parser_get_info = subparsers.add_parser("get-character-info", help="Get information about a character")
    parser_get_info.add_argument("character_name", type=str, help="Name of the character to query")

    args = parser.parse_args()

    if args.command == "compute-embeddings":
        compute_embeddings(args.stories_path)
    elif args.command == "get-character-info":
        get_character_info(args.character_name)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()








