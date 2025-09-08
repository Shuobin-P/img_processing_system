import streamlit as st

# Define the pages
main_page = st.Page("main_page.py", title="首页", icon="🎈")
preprocessing = st.Page("preprocessing.py", title="预处理", icon="❄️")
page_3 = st.Page("page_3.py", title="Page 3", icon="🎉")
page_4 = st.Page("page_4.py", title ="Page 4", icon ="🎉")

# Set up navigation
pg = st.navigation([main_page, preprocessing, page_3, page_4])

# Run the selected page
pg.run()