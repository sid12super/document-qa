import streamlit as st
from openai import OpenAI
import os

def main():
    # Show title and description.
    st.title("📄 Sid's Document Summarization (Lab 2)")
    st.write(
        "Upload a document below and select the type of summary you'd like to generate. "
        "You can also choose between the advanced model (4o) and the mini model (4o-mini)."
    )

    # Retrieve the OpenAI API key from Streamlit Secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_api_key:
        st.error("OpenAI API key is missing! Please configure it in Streamlit Secrets.")
        return

    # Sidebar options
    st.sidebar.title("Summary Options")
    summary_type = st.sidebar.radio(
        "Select Summary Type:",
        [
            "Summarize in 100 words",
            "Summarize in 2 connecting paragraphs",
            "Summarize in 5 bullet points",
        ],
    )
    use_advanced_model = st.sidebar.checkbox("Use Advanced Model (4o)")

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    if uploaded_file:
        # Read and decode the uploaded file.
        document = uploaded_file.read().decode()

        # Display the uploaded document.
        st.subheader("Uploaded Document")
        st.text_area("Document Content", document, height=300, disabled=True)

        # Determine the model to use.
        model = "gpt-4o" if use_advanced_model else "gpt-4o-mini"

        # Generate the summary based on the selected type.
        instructions = {
            "Summarize in 100 words": "Provide a concise summary of the document in 100 words.",
            "Summarize in 2 connecting paragraphs": "Provide a summary of the document in 2 connecting paragraphs.",
            "Summarize in 5 bullet points": "Provide a summary of the document in 5 bullet points.",
        }
        prompt = f"Document: {document}\n\nInstructions: {instructions[summary_type]}"

        # Create an OpenAI client.
        client = OpenAI(api_key=openai_api_key)

        try:
            # Generate the summary using the OpenAI API.
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response["choices"][0]["message"]["content"]

            # Display the summary.
            st.subheader("Generated Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
    else:
        st.info("Please upload a document to generate a summary.")

if __name__ == "__main__":
    main()