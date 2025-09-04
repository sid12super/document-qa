import streamlit as st
from openai import OpenAI
import os

def main():
    # Show title and description.
    st.title("📄 Sid's Document Bot - Lab 2")
    st.write("Upload a document and choose how you’d like it summarized!")

    # Retrieve the OpenAI API key from Streamlit Secrets or environment variable
    openai_api_key = (
    st.secrets["OPENAI_API_KEY"]
    if "OPENAI_API_KEY" in st.secrets
    else os.getenv("OPENAI_API_KEY")
    )

    if not openai_api_key:
        st.error("No API key found. Please set it in .streamlit/secrets.toml or as an environment variable.")
        return

    client = OpenAI(api_key=openai_api_key)

    # Sidebar: summary options
    st.sidebar.header("Summary Options")
    summary_type = st.sidebar.radio(
        "Choose summary style:",
        ("100 words", "2 paragraphs", "5 bullet points")
    )

    # Sidebar: model selection
    st.sidebar.header("Model Selection")
    model = st.sidebar.selectbox(
        "Choose the model:",
        options=["gpt-4o", "gpt-4o-mini", "gpt-5-chat-latest", "gpt-5-nano"],
        index=1  # default = gpt-4o-mini
    )

    # Upload document
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    if uploaded_file:
        # Read and decode the uploaded file
        document = uploaded_file.read().decode()

        # Prompt instruction based on summary type
        if summary_type == "100 words":
            instruction = "Summarize the document in about 100 words."
        elif summary_type == "2 paragraphs":
            instruction = "Summarize the document in exactly 2 connecting paragraphs."
        else:  # 5 bullet points
            instruction = "Summarize the document in 5 concise bullet points."

        # Prepare the prompt for the OpenAI API
        messages = [
            {
                "role": "user",
                "content": f"Here’s a document: {document}\n\n---\n\n{instruction}"
            }
        ]

        # Generate the summary
        with st.spinner(f"Generating summary using {model}..."):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )
                st.subheader("Generated Summary")
                st.write(response["choices"][0]["message"]["content"])
            except Exception as e:
                st.error(f"Error generating summary: {e}")
    else:
        st.info("Please upload a document to generate a summary.")

if __name__ == "__main__":
    main()