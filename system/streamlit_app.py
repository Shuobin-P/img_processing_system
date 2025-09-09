import streamlit as st

# Define the pages
main_page = st.Page("main_page.py", title="首页", icon="🎈")
preprocessing_page = st.Page("preprocessing.py", title="预处理", icon="❄️")
classification_page = st.Page("classification.py", title="分类与地物识别", icon="🎉")

# Set up navigation
pg = st.navigation([main_page, preprocessing_page, classification_page])

# Run the selected page
pg.run()