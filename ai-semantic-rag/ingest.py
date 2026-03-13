import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import endee

# 1. Load and Chunk PDF
def ingest_data(pdf_path):
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # 2. Setup Endee Client
    # Note: Replace with actual Endee connection details if required by their docs
    client = endee.Client() 
    collection = client.get_or_create_collection(name="academic_notes")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Insert into Vector DB
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk.page_content).tolist()
        collection.insert(
            ids=[f"id_{i}"],
            embeddings=[embedding],
            metadatas=[{"text": chunk.page_content}]
        )
    print(f"Successfully stored {len(chunks)} chunks in Endee Vector DB!")

if __name__ == "__main__":
    # Make sure you upload a file named 'notes.pdf' to your repo
    if os.path.exists("notes.pdf"):
        ingest_data("notes.pdf")
    else:
        print("Error: Please upload a file named 'notes.pdf' first.")
