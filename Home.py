import streamlit as st
# from streamlit_option_menu import option_menu
import sys
import streamlit_authenticator as stauth
import pickle
from pathlib import Path
import yaml
from yaml import SafeLoader
from streamlit_authenticator import Authenticate, authenticate
from streamlit_extras.switch_page_button import switch_page



st.set_page_config(
    page_title="Picanalytics",
    layout="wide",
    initial_sidebar_state='collapsed',   
)

a, b = st.columns([1, 10])

with a:
    st.image("img/logopicTRANSw.png", width=100)
with b:
    st.title("PICANALITIKA")





# selected = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal")
# selected

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('img/data.jpg') 


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("css/pica.css")

# hashed_passwords = stauth.Hasher(['123', '345']).generate()

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


# print (hashed_passwords)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

col1, col2 = st.columns([3.5, 1.5])
with col1:
    st.write('')
with col2:
    name, authentication_status, username = authenticator.login('Login', 'main')


    if authentication_status:
        switch_page('Dashboard')
        # authenticator.logout('Logout', 'main')
        # st.write(f'Welcome *{name}*')
        # st.title('Some content')
    elif authentication_status is False:
        st.error('Username/password is incorrect')
    # elif authentication_status is None:
        # st.warning('Please enter your username and password')


    # if st.session_state["authentication_status"]:
    #     switch_page('Dashboard')
    #     # placeholder1.add_bg_from_local('img/data.jpg')
    #     authenticator.logout('Logout', 'sidebar')
    #     st.write(f'Welcome *{st.session_state["name"]}*')
    #     st.title('Summary')
    # elif st.session_state["authentication_status"] == False:
    #     st.error('Username/password is incorrect')
    # # elif st.session_state["authentication_status"] == None:
    # #     st.warning('Please enter your username and password')
