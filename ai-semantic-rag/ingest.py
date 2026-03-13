import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- FIX: Local SDK Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../python-sdk")))

try:
    from endee_python_sdk import Client
except ImportError:
    print("Error: python-sdk folder not found.")

def run_ingestion():
    pdf_path = "notes.pdf"
    if not os.path.exists(pdf_path): return

    loader = PyPDFLoader(pdf_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())

    try:
        client = Client() 
        collection = client.get_or_create_collection(name="academic_notes")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk.page_content).tolist()
            collection.insert(ids=[f"id_{i}"], embeddings=[embedding], metadatas=[{"text": chunk.page_content}])
        print("✅ Ingestion Successful!")
    except Exception as e: print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_ingestion()
