import streamlit as st
import HW1  # Renamed from lab1
import HW2  # Renamed from lab2
import HW3  # Renamed from lab3

def main():
    st.set_page_config(page_title="HW Manager", page_icon="📚")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "HW1", "HW2", "HW3"])

    # Render the selected page
    if page == "Home":
        # Landing page content
        st.title("Welcome to the HW Manager")
        st.write(
            "This multi-page app allows you to explore different homework assignments. "
            "Use the sidebar to navigate to 'HW1', 'HW2', or 'HW3'!"
        )
    elif page == "HW1":
        HW1.main()  # Call the `main` function from HW1.py
    elif page == "HW2":
        HW2.main()  # Call the `main` function from HW2.py
    elif page == "HW3":
        HW3.main()  # Call the `main` function from HW3.py

if __name__ == "__main__":
    main()