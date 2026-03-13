import streamlit as st
import os
import sys

# --- FORCING STREAMLIT TO SEE THE SDK ---
# Streamlit Cloud mounts your repo at /mount/src/repo_name/
# We are manually adding that path to Python's memory
try:
    # This path is where Streamlit Cloud stores your files
    possible_path = "/mount/src/endee/python-sdk"
    if os.path.exists(possible_path):
        sys.path.insert(0, possible_path)
    
    # Also adding relative path as a backup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backup_path = os.path.abspath(os.path.join(current_dir, "../../python-sdk"))
    sys.path.insert(0, backup_path)

    # NOW TRY THE IMPORT
    from endee_python_sdk import Client
    from sentence_transformers import SentenceTransformer
    sdk_ready = True
except Exception as e:
    st.error(f"Critical Path Error: {e}")
    sdk_ready = False

st.set_page_config(page_title="Academic AI Search", layout="centered")
st.title("📚 Academic AI Assistant")

if sdk_ready:
    @st.cache_resource
    def load_resources():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = Client()
        return model, client

    model, client = load_resources()
    collection = client.get_or_create_collection(name="academic_notes")

    query = st.text_input("Ask a question from your notes:")
    if query:
        with st.spinner("Searching..."):
            query_vector = model.encode(query).tolist()
            results = collection.query(query_embeddings=[query_vector], n_results=1)
            if results['metadatas'] and len(results['metadatas'][0]) > 0:
                st.code(results['metadatas'][0][0]['text'], language="bash")
            else:
                st.warning("No match found.")
else:
    st.warning("Please check if 'python-sdk' folder is present in your GitHub root.")
