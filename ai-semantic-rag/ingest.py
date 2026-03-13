import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Link to local SDK
sys.path.append(os.path.abspath("../python-sdk"))
from endee_python_sdk import Client

def run_ingestion():
    pdf_path = "notes.pdf"
    if not os.path.exists(pdf_path):
        print("Error: notes.pdf not found!")
        return

    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 3. Initialize Endee Client and Model
    client = Client()
    collection = client.get_or_create_collection(name="academic_notes")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Store embeddings in Endee
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk.page_content).tolist()
        collection.insert(
            ids=[f"id_{i}"],
            embeddings=[embedding],
            metadatas=[{"text": chunk.page_content}]
        )
    print("✅ Successfully ingested notes into Endee DB!")

if __name__ == "__main__":
    run_ingestion()
