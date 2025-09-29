import streamlit as st
import openai
import os
import sys
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Fix for ChromaDB and Streamlit compatibility ---
# This workaround is necessary for ChromaDB to function correctly in environments like GitHub Codespaces.
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# --- Global Constants & Paths ---
CHROMA_DB_PATH = "./ChromaDB_RAG"
SOURCE_DIR = "su_orgs"  # Directory containing source HTML files
CHROMA_COLLECTION_NAME = "SU_Club_Collection"

# --- ChromaDB Client Initialization ---
# Initialize a persistent ChromaDB client to save the database to disk.
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
except Exception as e:
    st.error(f"Failed to initialize ChromaDB. Please check your setup. Error: {e}")
    st.stop()


# --- 1. Vector Database and Search Functions ---

def extract_text_from_html(file_path: str) -> str | None:
    """Extracts clean text content from an HTML file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        st.error(f"Error reading or parsing {os.path.basename(file_path)}: {e}")
        return None

def setup_vector_db(force_rebuild: bool = False):
    """
    Creates and populates the Vector DB from HTML files in the SOURCE_DIR.
    This function will only run if the database is empty or if a rebuild is forced.
    """
    if collection.count() > 0 and not force_rebuild:
        st.sidebar.info(f"Vector DB already contains {collection.count()} document chunks.")
        return

    st.sidebar.warning("Building Vector DB. This may take a moment...")
    with st.spinner("Processing files, chunking text, and creating embeddings..."):
        # Ensure the source directory exists
        if not os.path.exists(SOURCE_DIR):
            st.sidebar.error(f"Source directory '{SOURCE_DIR}' not found. Please create it and add your HTML files.")
            st.stop()
            
        source_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".html")]
        if not source_files:
            st.sidebar.error(f"No HTML files found in the '{SOURCE_DIR}' directory.")
            st.stop()

        openai_client = st.session_state.openai_client
        
        # Clear old collection if rebuilding
        if force_rebuild:
            chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
            global collection
            collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

        for filename in source_files:
            file_path = os.path.join(SOURCE_DIR, filename)
            doc_text = extract_text_from_html(file_path)
            if not doc_text:
                continue

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_text(doc_text)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i+1}"
                try:
                    response = openai_client.embeddings.create(input=chunk, model="text-embedding-3-small")
                    embedding = response.data[0].embedding
                    collection.add(documents=[chunk], ids=[chunk_id], embeddings=[embedding])
                except Exception as e:
                    st.error(f"Failed to embed chunk {chunk_id}: {e}")
    
    st.sidebar.success(f"Vector DB built successfully with {collection.count()} chunks.", icon="✅")

def get_relevant_club_info(query: str, n_results: int = 3) -> str:
    """
    Takes a user query, embeds it, and performs a vector search in ChromaDB 
    to find the most relevant text chunks (course/club info).
    """
    try:
        openai_client = st.session_state.openai_client
        query_response = openai_client.embeddings.create(input=[query], model="text-embedding-3-small")
        query_embedding = query_response.data[0].embedding
        
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        
        context = "\n---\n".join(results['documents'][0]) if results.get('documents') else "No relevant information found in the documents."
        return context
    except Exception as e:
        st.error(f"Error querying Vector DB: {e}")
        return "Error retrieving context from the database."


# --- 2. LLM Invocation ---

def get_openai_response(user_prompt: str, context: str, chat_history: list, model_name: str) -> str:
    """
    Invokes the OpenAI LLM with the user's prompt, retrieved context, and conversation history.
    """
    system_prompt = f"""
    You are an expert assistant for Syracuse University student clubs. Your task is to answer the user's question based on the provided context and conversation history.
    - Use the 'RELEVANT INFORMATION' to ground your answer.
    - Use the 'CONVERSATION HISTORY' to understand the flow of the conversation.
    - If the answer is in the provided information, synthesize it. Do not just copy-paste.
    - If the answer is not found, clearly state that you couldn't find the information in your documents.
    - Keep your answers helpful and conversational.
    
    RELEVANT INFORMATION:
    {context}
    """
    
    # Combine system prompt with user prompt and history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    try:
        client = st.session_state.openai_client
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        return "Sorry, I encountered an error while generating a response."

# --- 3. The Main Streamlit App ---

def main():
    st.title("🧠 SU Club Information Chatbot")
    st.write("Ask me questions about student organizations at Syracuse University!")

    # --- Sidebar for Configuration and DB Management ---
    with st.sidebar:
        st.header("Configuration")
        selected_model = st.selectbox(
            "Choose an OpenAI Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )
        st.markdown("---")
        st.header("Database Management")
        if st.button("Re-Build Vector DB"):
            setup_vector_db(force_rebuild=True)
            st.rerun()

    # --- Initialize API client and Vector DB ---
    try:
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        setup_vector_db()
    except KeyError:
        st.error("`OPENAI_API_KEY` not found. Please set it in your Streamlit secrets.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {e}")
        st.stop()

    # --- Chat History Management ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Main Chat Logic ---
    if prompt := st.chat_input("Ask about SU clubs..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Perform vector search to get relevant info
                relevant_info = get_relevant_club_info(query=prompt)
                
                # 2. Get the last 5 turns of conversation history
                chat_history = st.session_state.messages[-11:-1]

                # 3. Invoke the LLM with context and history
                response_content = get_openai_response(
                    user_prompt=prompt,
                    context=relevant_info,
                    chat_history=chat_history,
                    model_name=selected_model
                )
                
                st.markdown(response_content)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    main()