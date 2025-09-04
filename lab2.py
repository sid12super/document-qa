import streamlit as st

def main():
    # Show title and description.
    st.title("Lab2 Page")
    st.write("Welcome to Lab2! This page is under development.")
    st.write("Feel free to explore and add new features.")

    # Retrieve the LLM key from Streamlit Secrets
    llm_key = st.secrets.get("OPENAI_API_KEY")
    if not llm_key:
        st.error("LLM key is missing! Please configure it in Streamlit Secrets.")
    else:
        st.write("LLM key successfully retrieved from Streamlit Secrets.")

if __name__ == "__main__":
    main()