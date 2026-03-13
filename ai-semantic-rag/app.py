import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Client

st.set_page_config(page_title="Academic AI Assistant", layout="centered")

st.title("📚 Academic AI Assistant")
st.write("Search through your notes using AI-powered semantic search.")

# Initialize Model and Client
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = Client()
    return model, client

try:
    model, client = load_resources()
    collection = client.get_collection(name="academic_notes")

    # Search UI
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
    st.error(f"Setup Error: {e}")
    st.info("Make sure you have run ingest.py first to store data.")
