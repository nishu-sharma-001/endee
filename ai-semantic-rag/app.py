import streamlit as st
import os
import sys

# --- SDK FOLDER CONNECTION ---
# Define the current directory and find the path to the 'python-sdk' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
# Note: This looks for the sdk folder two levels up from the current app script
sdk_path = os.path.abspath(os.path.join(current_dir, "../../python-sdk"))

# Add the SDK path to sys.path so Python can find the local package
if os.path.exists(sdk_path):
    sys.path.append(sdk_path)
    try:
        # Importing the Client directly from the local folder
        from endee_python_sdk import Client 
        sdk_ready = True
    except ImportError:
        sdk_ready = False
else:
    sdk_ready = False

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Academic AI Assistant", layout="centered")
st.title("📚 Academic AI Assistant")

if not sdk_ready:
    st.error(f"❌ SDK folder not found or import failed. Path checked: {sdk_path}")
    st.info("Check if 'python-sdk' folder exists in your GitHub repository.")
else:
    # Import heavy libraries only if the SDK is successfully linked
    from sentence_transformers import SentenceTransformer
    
    # Cache resources to prevent reloading the model on every user interaction
    @st.cache_resource
    def load_resources():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = Client()
        return model, client

    try:
        # Initialize the AI model and Endee client
        model, client = load_resources()
        collection = client.get_or_create_collection(name="academic_notes")
        
        # User input for semantic search
        query = st.text_input("Ask a question from your notes (e.g., How to find factorial?):")
        
        if query:
            with st.spinner("Searching through notes..."):
                # Convert the user's question into a mathematical vector (embedding)
                query_vector = model.encode(query).tolist()
                
                # Query the Endee database for the most relevant chunk
                results = collection.query(query_embeddings=[query_vector], n_results=1)
                
                # Check if we found a valid match in the metadata
                if results['metadatas'] and len(results['metadatas'][0]) > 0:
                    st.subheader("Found in Notes:")
                    # Display the retrieved code/text block
                    st.code(results['metadatas'][0][0]['text'], language="bash")
                else:
                    st.warning("No relevant match found in your notes.")
                    
    except Exception as e:
        st.info("💡 Hint: Ensure you have run the ingestion script (ingest.py) to store data in the database.")
