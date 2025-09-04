import streamlit as st
from openai import OpenAI

def main():
    # Show title and description.
    st.title("📄 Lab 2B")
    st.write(
        "Upload a document below and ask a question about it – GPT will answer! "
        "To use this app, you need to configure your OpenAI API key in Streamlit Secrets. "
        "Using GPT-5-nano"
    )

    # Retrieve the LLM key from Streamlit Secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key is missing! Please configure it in Streamlit Secrets.")
    else:
        # Create an OpenAI client.
        client = OpenAI(api_key=openai_api_key)

        # Let the user upload a file via `st.file_uploader`.
        uploaded_file = st.file_uploader(
            "Upload a document (.txt or .md)", type=("txt", "md")
        )

        # Ask the user for a question via `st.text_area`.
        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
        )

        if uploaded_file and question:
            # Process the uploaded file and question.
            document = uploaded_file.read().decode()
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)

if __name__ == "__main__":
    main()