import streamlit as st
import lab1  # Import lab1.py to reuse its functionality

# Set page config
st.set_page_config(page_title="Lab Manager", page_icon=":material/apps:")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Lab 2 (Default)", "Lab 1"])

# Render the selected page
if page == "Lab 2 (Default)":
    # Show title and description.
    st.title("Lab2 Page")
    st.write("Welcome to Lab2! This page is under development.")
    st.write("Feel free to explore and add new features.")
elif page == "Lab 1":
    lab1.main()  # Call a `main` function from lab1.py