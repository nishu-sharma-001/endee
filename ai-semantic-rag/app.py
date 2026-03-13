import streamlit as st
import os
import sys

# --- FORCING THE PATH ---
# Streamlit Cloud mounts the repo at /mount/src/endee/
# We need to point exactly to where the python-sdk is.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Going up to project root and then into python-sdk
sdk_path = os.path.abspath(os.path.join(current_dir, "../../python-sdk"))

if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

try:
    # After adding __init__.py, this should work
    from endee_python_sdk import Client 
    from sentence_transformers import SentenceTransformer
    sdk_ready = True
except Exception as e:
    st.error(f"Import Error: {e}")
    sdk_ready = False

st.set_page_config(page_title="Academic AI Assistant", layout="centered")
st.title("📚 Academic AI Assistant")

if sdk_ready:
    @st.cache_resource
    def load_resources():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = Client()
        return model, client

    model, client = load_resources()
    # Your search logic here...
    query = st.text_input("Ask a question from your notes:")
    if query:
        st.write(f"Searching for: {query}")
        # (Add your search results logic here)
