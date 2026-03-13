import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- SDK PATH INJECTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sdk_path = os.path.abspath(os.path.join(current_dir, "../../python-sdk"))
sys.path.append(sdk_path)

try:
    from endee_python_sdk import Client
except ImportError:
    print("❌ SDK Folder not found at:", sdk_path)

def run_ingestion():
    pdf_path = "notes.pdf"
    if not os.path.exists(pdf_path):
        print("❌ notes.pdf missing!")
        return

    print("🔄 Processing PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    try:
        client = Client() 
        collection = client.get_or_create_collection(name="academic_notes")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk.page_content).tolist()
            collection.insert(
                ids=[f"id_{i}"],
                embeddings=[embedding],
                metadatas=[{"text": chunk.page_content}]
            )
        print("✅ SUCCESS: Data stored in Endee DB!")
    except Exception as e:
        print(f"❌ Storage Error: {e}")

if __name__ == "__main__":
    run_ingestion()
