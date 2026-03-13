import streamlit as st
import os
import sys

# --- FIX: Manually link to the SDK folder ---
# Hum check kar rahe hain ki 'python-sdk' folder kahan hai
current_dir = os.path.dirname(os.path.abspath(__file__))
sdk_path = os.path.abspath(os.path.join(current_dir, "../../python-sdk"))
if sdk_path not in sys.path:
    sys.path.append(sdk_path)

try:
    from endee_python_sdk import Client # Import directly from the folder
    from sentence_transformers import SentenceTransformer
    st.success("✅ Endee SDK Linked Successfully!")
except ImportError:
    st.error("❌ Could not find endee_python_sdk folder. Check your repo structure.")

st.set_page_config(page_title="Academic AI Assistant", layout="centered")
st.title("📚 Academic AI Assistant")

# Initialize Model and Client
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = Client()
    return model, client

try:
    model, client = load_resources()
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
    st.info("Ingest your notes first to see results here.")
