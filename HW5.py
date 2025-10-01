# HW5.py

import streamlit as st
import openai
import google.generativeai as genai
from anthropic import Anthropic
import os
import sys
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Fix for ChromaDB and Streamlit compatibility ---
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# --- Global Constants & Paths ---
CHROMA_DB_PATH = "./ChromaDB_RAG"
SOURCE_DIR = "su_orgs"  # Directory containing source HTML files
CHROMA_COLLECTION_NAME = "SU_Club_Collection"

# --- ChromaDB Client Initialization ---
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
    """
    global collection

    if collection.count() > 0 and not force_rebuild:
        st.sidebar.info(f"Vector DB contains {collection.count()} chunks.")
        return

    st.sidebar.warning("Building Vector DB...")
    with st.spinner("Processing files, chunking text, and creating embeddings..."):
        if not os.path.exists(SOURCE_DIR):
            st.sidebar.error(f"Source directory '{SOURCE_DIR}' not found. Please add your HTML files.")
            st.stop()
            
        source_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".html")]
        if not source_files:
            st.sidebar.error(f"No HTML files found in the '{SOURCE_DIR}' directory.")
            st.stop()

        if 'openai_client' not in st.session_state:
             st.session_state.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        openai_client = st.session_state.openai_client
        
        if force_rebuild:
            chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
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
    
    st.sidebar.success(f"Vector DB built with {collection.count()} chunks.", icon="✅")
    
def get_relevant_club_info(query: str, n_results: int = 3) -> str:
    """
    Performs a vector search in ChromaDB to find relevant text chunks.
    """
    try:
        openai_client = st.session_state.openai_client
        query_response = openai_client.embeddings.create(input=[query], model="text-embedding-3-small")
        query_embedding = query_response.data[0].embedding
        
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        
        context = "\n---\n".join(results['documents'][0]) if results.get('documents') else "No relevant information found."
        return context
    except Exception as e:
        st.error(f"Error querying Vector DB: {e}")
        return "Error retrieving context from the database."


# --- 2. LLM Invocation ---

def get_llm_response(provider: str, model: str, user_prompt: str, context: str, chat_history: list) -> str:
    """
    Invokes the selected LLM with the user's prompt, context, and history.
    """
    system_prompt = f"""
    You are an expert assistant for Syracuse University student clubs. Your task is to answer the user's question based on the provided context and conversation history.
    - Use the 'RELEVANT INFORMATION' to ground your answer.
    - If the answer is not found, state that you couldn't find the information in your documents.
    - Keep your answers helpful and conversational.
    
    RELEVANT INFORMATION:
    {context}
    """
    
    try:
        if provider == "OpenAI":
            client = st.session_state.openai_client
            messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_prompt}]
            response = client.chat.completions.create(model=model, messages=messages, max_completion_tokens=1024)
            return response.choices[0].message.content

        elif provider == "Gemini":
            client = st.session_state.gemini_client
            # For Gemini, combine history and prompts into a single string
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
            full_prompt = f"{system_prompt}\n\nCONVERSATION HISTORY:\n{history_str}\n\nUSER QUESTION:\n{user_prompt}"
            response = client.generate_content(full_prompt)
            return response.text

        elif provider == "Anthropic":
            client = st.session_state.anthropic_client
            response = client.messages.create(
                model=model,
                system=system_prompt,  # Use the dedicated system prompt parameter
                messages=chat_history + [{"role": "user", "content": user_prompt}],
                max_tokens=1024
            )
            return response.content[0].text
            
    except Exception as e:
        st.error(f"An error occurred with {provider}: {e}")
        return f"Sorry, I encountered an error with the {provider} API."


# --- 3. The Main Streamlit App ---

def main():
    st.title("🧠 SU Club Information Chatbot")
    st.write("Ask questions about student organizations at Syracuse University!")

    # --- Sidebar for Configuration and DB Management ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LLM Provider Selection
        selected_provider = st.selectbox(
            "Choose an LLM Provider",
            ["OpenAI", "Gemini", "Anthropic"]
        )

        # Model options based on provider
        model_options = {
            "OpenAI": ["gpt-5-nano", "gpt-5-chat-latest"],
            "Gemini": ["gemini-2.5-pro", "gemini-2.5-flash"],
            "Anthropic": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"]
        }
        
        # Model Selection
        selected_model = st.selectbox(
            "Choose a Model",
            model_options[selected_provider]
        )
        
        st.markdown("---")
        st.header("🗄️ Database Management")
        if st.button("Re-Build Vector DB"):
            setup_vector_db(force_rebuild=True)
            st.rerun()

    # --- Initialize API clients and Vector DB ---
    try:
        # Store clients in session state to avoid re-initializing
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        if 'gemini_client' not in st.session_state:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            st.session_state.gemini_client = genai.GenerativeModel(selected_model) # Model name for Gemini is needed here
        if 'anthropic_client' not in st.session_state:
            st.session_state.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        
        # Build the DB on first run
        setup_vector_db()

    except KeyError as e:
        st.error(f"API Key Error: Please make sure `{e.args[0]}` is set in your Streamlit secrets.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during initialization: {e}")
        st.stop()

    # --- Chat History Management ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Main Chat Logic ---
    if prompt := st.chat_input("Ask about SU clubs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Asking {selected_model}..."):
                relevant_info = get_relevant_club_info(query=prompt)
                chat_history = st.session_state.messages[-11:-1] # Last 5 turns

                response_content = get_llm_response(
                    provider=selected_provider,
                    model=selected_model,
                    user_prompt=prompt,
                    context=relevant_info,
                    chat_history=chat_history
                )
                
                st.markdown(response_content)
        
        st.session_state.messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    main()