import streamlit as st
import lab1  # Import lab1.py for Document QA functionality
import lab2  # Import lab2.py for Lab2 functionality

def main():
    st.set_page_config(page_title="Multi-page App", page_icon="📄")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Landing Page", "Lab1", "Lab2"])

    # Render the selected page
    if page == "Landing Page":
        # Landing page content
        st.title("Welcome to Sid's Multi-Page App")
        st.write(
            "This multi-page app allows you to explore different functionalities. "
            "Navigate to the 'Document QA' or 'Lab2' page to explore more!"
        )
        st.image(
            "https://via.placeholder.com/800x400.png?text=Document+QA+App",
            caption="Sid's Document QA App",
        )
    elif page == "Lab1":
        lab1.main()  # Call the `main` function from lab1.py
    elif page == "Lab2":
        lab2.main()  # Call the `main` function from lab2.py

if __name__ == "__main__":
    main()