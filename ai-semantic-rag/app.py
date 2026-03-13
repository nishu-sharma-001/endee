import streamlit as st
import os
import sys
from sentence_transformers import SentenceTransformer

# Link to the local SDK folder (Relative path for Codespaces)
sys.path.append(os.path.abspath("../python-sdk"))

try:
    from endee_python_sdk import Client
    sdk_ready = True
except ImportError:
    sdk_ready = False

st.title("📚 Academic AI Search")

if sdk_ready:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = Client()
    collection = client.get_or_create_collection(name="academic_notes")

    query = st.text_input("Search your notes (e.g., Factorial code):")
    if query:
        query_vector = model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_vector], n_results=1)
        
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            st.code(results['metadatas'][0][0]['text'])
else:
    st.error("SDK not found. Make sure you are in the ai-semantic-rag folder.")
