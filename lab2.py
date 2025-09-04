import streamlit as st

# Define the pages
lab2_page = st.Page("lab2.py", title="Lab 2 (Default)", icon=":material/home:")
lab1_page = st.Page("lab1.py", title="Lab 1", icon=":material/science:")

# Setup navigation
pg = st.navigation([lab2_page, lab1_page])

# Set page config
st.set_page_config(page_title="Lab Manager", page_icon=":material/apps:")

# Run the navigation
pg.run()