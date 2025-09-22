import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import os
from PyPDF2 import PdfReader
import sys

# --- Fix for working with ChromaDB and Streamlit ---
# This is a workaround for a known issue with ChromaDB and Streamlit's environment.
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# --- Global Constants & Paths ---
CHROMA_DB_PATH = "./ChromaDB_RAG"
PDF_SOURCE_DIR = "su_orgs"
CHROMA_COLLECTION_NAME = "MultiDocCollection"

# Initialize ChromaDB client. It's persistent, so it saves to disk.
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# --- Helper Functions ---

def extract_text_from_pdf(file_path):
    """Extracts all text from a given PDF file."""
    try:
        pdf_reader = PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text
    except Exception as e:
        st.error(f"Error extracting text from {os.path.basename(file_path)}: {e}")
        return None

def setup_vector_db():
    """
    Creates the Vector DB once if it doesn't already exist.
    It processes all PDFs in the source directory, chunks them, and embeds each chunk.
    """
    # Requirement 2a: Create the Vector DB once if the file does not exist.
    # We check if the collection is empty. If it's not, we assume it's been built.
    if collection.count() > 0:
        st.sidebar.info(f"Vector DB already contains {collection.count()} document chunks.")
        return

    st.sidebar.warning("Vector DB is empty. Building it now... Please wait.")
    with st.spinner("Processing PDFs, chunking, and creating embeddings..."):
        pdf_files = [f for f in os.listdir(PDF_SOURCE_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            st.sidebar.error(f"No PDF files found in '{PDF_SOURCE_DIR}' directory.")
            st.stop()

        openai_client = st.session_state.openai_client
        
        for filename in pdf_files:
            file_path = os.path.join(PDF_SOURCE_DIR, filename)
            doc_text = extract_text_from_pdf(file_path)
            if not doc_text:
                continue

            # Requirement 1c & 2: "Chunk" each document into two mini-documents.
            # --- Chunking Method Explanation ---
            # I am using a simple 50/50 split method for chunking.
            # Why this method?
            # 1. Simplicity & Speed: It's computationally inexpensive and fast to implement.
            # 2. Fulfills Requirement: It directly addresses the request for "two separate mini-documents".
            # 3. Context Preservation: For many documents, a straight split can keep related paragraphs
            #    together better than more complex methods like fixed-size chunks, which can cut sentences
            #    in half. This method ensures each chunk is a substantial, contiguous block of text.
            midpoint = len(doc_text) // 2
            chunks = [doc_text[:midpoint], doc_text[midpoint:]]

            # Embed and add each chunk to the collection
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
    
    st.sidebar.success(f"Vector DB created successfully with {collection.count()} document chunks.", icon="✅")


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
        # Combine the retrieved document chunks into a single context string
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
    
    # Prepare a unified prompt for the LLM
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
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": final_prompt}
                    ],
                    max_tokens=2048,
                )
                return response.choices[0].message.content

            elif llm_provider == "Google":
                client = st.session_state.gemini_client
                full_prompt_for_gemini = system_prompt + "\n" + final_prompt
                response = client.generate_content(full_prompt_for_gemini)
                return response.text

            elif llm_provider == "Anthropic":
                client = st.session_state.anthropic_client
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    messages=[{"role": "user", "content": final_prompt}],
                    max_tokens=2048,
                )
                return response.content[0].text
                
    except Exception as e:
        st.error(f"An error occurred with {llm_provider}: {e}")
        return f"Sorry, I encountered an error while contacting {llm_provider}."

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Multi-LLM RAG Chat", page_icon="🧠")
    st.title("🧠 Multi-LLM RAG Chat Application")
    st.write("Ask questions about documents in the 'su_orgs' folder.")

    # --- Sidebar for Settings ---
    st.sidebar.header("Settings")

    # Initialize LLM Clients safely using st.secrets
    try:
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        if 'gemini_client' not in st.session_state:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            st.session_state.gemini_client = genai.GenerativeModel('gemini-1.5-pro-latest')
        if 'anthropic_client' not in st.session_state:
            st.session_state.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except Exception as e:
        st.error(f"Failed to initialize one or more LLM clients. Please check your API keys in Streamlit secrets. Error: {e}")
        st.stop()

    # Requirement 2e: Let the user pick between 3 different LLMs.
    selected_llm = st.sidebar.selectbox(
        "Choose an LLM:",
        ("OpenAI", "Google", "Anthropic"),
        help="Select the language model you want to use for answering."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Vector Database")
    # Requirement 1 & 2a: Create the Vector DB once.
    setup_vector_db()

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question about the documents..."):
        # Add user message to state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 1. Augment: Get relevant context from the Vector DB
        context = query_vector_db(prompt)

        # 2. Memory: Create the conversational memory buffer
        # Requirement 2c: Store up to 5 questions and answers.
        # We'll take the last 10 messages (5 user + 5 assistant).
        chat_history = st.session_state.messages[-11:-1]

        # 3. Generate: Get the final response from the selected LLM
        response_content = get_llm_response(selected_llm, prompt, context, chat_history)
        
        # Add assistant response to state and display it
        full_response = f"**Answer from {selected_llm}:**\n\n{response_content}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        with st.chat_message("assistant"):
            st.markdown(full_response)

if __name__ == "__main__":
    main()
