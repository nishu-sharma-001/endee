import streamlit as st
from sentence_transformers import SentenceTransformer
import endee

st.set_page_config(page_title="Academic AI Search", page_icon="📚")
st.title("📚 Academic Semantic Search")
st.markdown("Powered by **Endee Vector Database**")

model = SentenceTransformer('all-MiniLM-L6-v2')
client = endee.Client()
collection = client.get_collection(name="academic_notes")

query = st.text_input("Ask a question from your notes:")

if query:
    with st.spinner("Searching..."):
        # Convert query to vector
        query_vector = model.encode(query).tolist()
        
        # Search in Endee
        results = collection.query(query_embeddings=[query_vector], n_results=3)
        
        st.subheader("Relevant Results:")
        for res in results['metadatas'][0]:
            st.write(f"📖 {res['text']}")
            st.divider()
