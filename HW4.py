import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import os
from bs4 import BeautifulSoup  # Import BeautifulSoup for HTML parsing
import sys

# --- Fix for working with ChromaDB and Streamlit ---
# This is a workaround for a known issue with ChromaDB and Streamlit's environment.
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Global Constants & Paths ---
CHROMA_DB_PATH = "./ChromaDB_RAG"
SOURCE_DIR = "su_orgs"  # Directory containing source files
CHROMA_COLLECTION_NAME = "MultiDocCollection"

# Initialize ChromaDB client. It's persistent, so it saves to disk.
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# --- Helper Functions ---

def extract_text_from_html(file_path):
    """Extracts all text from a given HTML file, stripping out tags."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            # Use .get_text() to extract clean text from all HTML tags
            text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        st.error(f"Error extracting text from {os.path.basename(file_path)}: {e}")
        return None

def setup_vector_db(force_rebuild=False):
    """
    Creates the Vector DB once if it doesn't already exist or if a rebuild is forced.
    It processes all HTML files, chunks them using Langchain, and embeds each chunk.
    """
    if collection.count() > 0 and not force_rebuild:
        st.sidebar.info(f"Vector DB already contains {collection.count()} document chunks.")
        return

    st.sidebar.warning("Vector DB is being built. Please wait...")
    with st.spinner("Processing HTML files, chunking, and creating embeddings..."):
        # UPDATED: Look for .html files instead of .pdf
        source_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".html")]
        if not source_files:
            st.sidebar.error(f"No HTML files found in '{SOURCE_DIR}' directory.")
            st.stop()

        openai_client = st.session_state.openai_client
        
        for filename in source_files:
            file_path = os.path.join(SOURCE_DIR, filename)
            # UPDATED: Use the new HTML extraction function
            doc_text = extract_text_from_html(file_path)
            if not doc_text:
                continue

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(doc_text)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i+1}"
                try:
                    response = openai_client.embeddings.create(
                        input=chunk,
                        model="text-embedding-3-small"
                    )
                    embedding = response.data[0].embedding
                    collection.add(
                        documents=[chunk],
                        ids=[chunk_id],
                        embeddings=[embedding]
                    )
                except Exception as e:
                    st.error(f"Failed to embed chunk {chunk_id}: {e}")
    
    st.sidebar.success(f"Vector DB built successfully with {collection.count()} document chunks.", icon="✅")


def rebuild_vector_db():
    """Deletes the existing collection and rebuilds it from the source files."""
    with st.spinner("Deleting existing Vector DB and rebuilding... Please wait."):
        try:
            chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
            global collection
            collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
            setup_vector_db(force_rebuild=True)
            st.sidebar.success("Vector DB has been successfully rebuilt!", icon="🔄")
        except Exception as e:
            st.sidebar.error(f"Error while rebuilding Vector DB: {e}")

def query_vector_db(prompt, n_results=4):
    """Queries the vector database to find relevant document chunks for the user's prompt."""
    try:
        openai_client = st.session_state.openai_client
        query_response = openai_client.embeddings.create(input=[prompt], model="text-embedding-3-small")
        query_embedding = query_response.data[0].embedding
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        context = "\n---\n".join(results['documents'][0]) if results.get('documents') else "No relevant context found."
        return context
    except Exception as e:
        st.error(f"Error querying Vector DB: {e}")
        return "Error retrieving context from the database."

def get_llm_response(llm_provider, prompt, context, chat_history):
    """
    Calls the selected LLM with the prompt, context, and chat history
    to generate a final response.
    """
    system_prompt = f"""
    You are an expert assistant. Your task is to answer the user's question based on the provided context and conversation history.
    - Use the 'CONTEXT FROM DOCUMENTS' to ground your answer.
    - Use the 'CONVERSATION HISTORY' to understand the flow of the conversation.
    - If the answer is in the context, synthesize it. Do not just copy-paste.
    - If the answer is not in the context but is in the conversation history, use the history.
    - If the answer is not found in either, clearly state that you couldn't find the information in the provided documents or history.
    - Keep your answers concise and to the point.
    """
    
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    final_prompt = f"""
    CONTEXT FROM DOCUMENTS:
    {context}

    CONVERSATION HISTORY:
    {history_str}

    USER'S QUESTION:
    {prompt}
    """

    try:
        with st.spinner(f"Asking {llm_provider}..."):
            if llm_provider == "OpenAI":
                client = st.session_state.openai_client
                response = client.chat.completions.create(model="gpt-5-chat-latest", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": final_prompt}], max_tokens=2048)
                return response.choices[0].message.content
            elif llm_provider == "Google":
                client = st.session_state.gemini_client
                full_prompt_for_gemini = system_prompt + "\n" + final_prompt
                response = client.generate_content(full_prompt_for_gemini)
                return response.text
            elif llm_provider == "Anthropic":
                client = st.session_state.anthropic_client
                response = client.messages.create(model="claude-sonnet-4-20250514", system=system_prompt, messages=[{"role": "user", "content": final_prompt}], max_tokens=2048)
                return response.content[0].text
    except Exception as e:
        st.error(f"An error occurred with {llm_provider}: {e}")
        return f"Sorry, I encountered an error while contacting {llm_provider}."

def main():
    st.set_page_config(page_title="Multi-LLM RAG Chat", page_icon="🧠")
    st.title("🧠 Multi-LLM RAG Chat Application")
    st.write(f"Ask questions about documents in the '{SOURCE_DIR}' folder.")

    st.sidebar.header("Settings")
    try:
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        if 'gemini_client' not in st.session_state:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            st.session_state.gemini_client = genai.GenerativeModel('gemini-2.5-pro-latest')
        if 'anthropic_client' not in st.session_state:
            st.session_state.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except Exception as e:
        st.error(f"Failed to initialize one or more LLM clients. Please check your API keys in Streamlit secrets. Error: {e}")
        st.stop()

    selected_llm = st.sidebar.selectbox("Choose an LLM:", ("OpenAI", "Google", "Anthropic"))
    
    st.sidebar.markdown("---")
    st.sidebar.header("Management")
    if st.sidebar.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("Re-Build Vector DB", key="rebuild_db"):
        rebuild_vector_db()
        st.rerun()
        
    st.sidebar.markdown("---")
    setup_vector_db()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = query_vector_db(prompt)
        chat_history = st.session_state.messages[-11:-1]
        response_content = get_llm_response(selected_llm, prompt, context, chat_history)
        
        full_response = f"**Answer from {selected_llm}:**\n\n{response_content}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        with st.chat_message("assistant"):
            st.markdown(full_response)

if __name__ == "__main__":
    main()

