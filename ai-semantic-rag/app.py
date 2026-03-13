import streamlit as st
import os
import sys

# --- SDK PATH INJECTION ---
# Hum folder-by-folder dhoond rahe hain taaki 'ModuleNotFoundError' na aaye
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sdk_path = os.path.join(project_root, "python-sdk")

if os.path.exists(sdk_path):
    sys.path.append(sdk_path)
    try:
        from endee_python_sdk import Client
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        st.error(f"Folder mil gaya par import nahi hua: {e}")
else:
    st.error(f"SDK folder nahi mila! Path: {sdk_path}")

st.set_page_config(page_title="Academic AI Assistant", layout="centered")
st.title("📚 Academic AI Assistant")

# Model Load logic
@st.cache_resource
def load_resources():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = Client()
        return model, client
    except NameError:
        return None, None

model, client = load_resources()

if model and client:
    try:
        collection = client.get_or_create_collection(name="academic_notes")
        query = st.text_input("Ask a question from your notes (e.g., How to find factorial?):")

        if query:
            with st.spinner("Searching..."):
                query_vector = model.encode(query).tolist()
                results = collection.query(query_embeddings=[query_vector], n_results=1)

                if results['metadatas'] and len(results['metadatas'][0]) > 0:
                    st.subheader("Match Found:")
                    st.code(results['metadatas'][0][0]['text'], language="bash")
                else:
                    st.warning("No relevant content found in notes.")
    except Exception as e:
        st.info("💡 Hint: Pehle 'ingest.py' run karke data save karein.")
else:
    st.warning("⚠️ Setup incomplete. Please check logs.")
