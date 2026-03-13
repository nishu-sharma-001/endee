import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from endee import Client  # Standard import

def run_ingestion():
    # PDF path (Ensure notes.pdf is in the same folder)
    pdf_path = "notes.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
        return

    # 1. Load and Split PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks.")

    # 2. Setup Endee Client
    try:
        client = Client() 
        collection = client.get_or_create_collection(name="academic_notes")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 3. Store Vectors
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk.page_content).tolist()
            collection.insert(
                ids=[f"id_{i}"],
                embeddings=[embedding],
                metadatas=[{"text": chunk.page_content}]
            )
        print("✅ Data Ingestion Successful!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_ingestion()
