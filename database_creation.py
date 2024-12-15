from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv(".env")

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API')

pc = Pinecone()

index_name = "asign_db"

if index_name not in pc.list_indexes():
  try:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Database Created")
  except:
    print("Already exist")