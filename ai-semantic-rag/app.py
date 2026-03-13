import streamlit as st
import os
import sys
from sentence_transformers import SentenceTransformer

# Adding the local SDK folder to the system path
sys.path.append(os.path.abspath("../python-sdk"))

try:
    from endee_python_sdk import Client
    sdk_ready = True
except ImportError:
    sdk_ready = False

# --- UI Setup ---
st.set_page_config(page_title="Academic AI Assistant", page_icon="📚")
st.title("📚 Academic AI Assistant")
st.markdown("Search through your academic notes using Semantic AI.")

if sdk_ready:
    # Initialize Model and Client
    @st.cache_resource
    def load_resources():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = Client()
        return model, client

    try:
        model, client = load_resources()
        collection = client.get_or_create_collection(name="academic_notes")

        # User Query Input
        query = st.text_input("Ask a question (e.g., How to find factorial?):")

        if query:
            with st.spinner("Searching..."):
                # Convert query to vector
                query_vector = model.encode(query).tolist()
                # Search in Endee Database
                results = collection.query(query_embeddings=[query_vector], n_results=1)

                if results['metadatas'] and len(results['metadatas'][0]) > 0:
                    st.subheader("Match Found in Notes:")
                    st.code(results['metadatas'][0][0]['text'], language="bash")
                else:
                    st.warning("No relevant content found in your notes.")
    except Exception as e:
        st.info("💡 Tip: Make sure to run 'ingest.py' first to process your notes.")
else:
    st.error("Endee SDK not detected. Please check your folder structure.")
