import streamlit as st
import PyPDF2
from openai import OpenAI

# Show title and description.
st.title("📄 Sid's Document Question Answering")

# Add a dropdown for model selection.
model = st.selectbox(
    "Select a GPT model",
    options=["gpt-3.5-turbo", "gpt-4.1", "gpt-5-nano", "gpt-5-chat-latest"],
    index=2,  # Default to "gpt-5-nano"
)

# Display the selected model dynamically.
st.write(
    f"Upload a document below and ask a question about it – GPT will answer! "
    f"To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    f"Using model {model}"
)

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Process the uploaded file and question.
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                document = ""
                for page in pdf_reader.pages:
                    document += page.extract_text()
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
                document = None
        else:
            st.error("Unsupported file type.")
            document = None

        if document:
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model=model,  # Use the selected model
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)
