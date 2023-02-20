import streamlit as st
# from streamlit_option_menu import option_menu
import sys


st.set_page_config(
    page_title="Picanalytics",
    layout="wide",
    initial_sidebar_state='expanded',   
)

# selected = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal")
# selected

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)