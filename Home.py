import streamlit as st
import os

# from streamlit_option_menu import option_menu
import sys
import streamlit_authenticator as stauth
import pickle
from pathlib import Path
import yaml
from yaml.loader import SafeLoader

# from streamlit_authenticator import Authenticate, authenticate
from streamlit_extras.switch_page_button import switch_page

from django.core.wsgi import get_wsgi_application
# from django.contrib.auth import authenticate


st.set_page_config(
    page_title="Forteks",
    layout="wide",
    initial_sidebar_state="collapsed",
)

a, b = st.columns([1, 10])

with a:
    st.image("img/logopicTRANSw.png", width=100)
with b:
    st.title("FORTEKS")


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
        unsafe_allow_html=True,
    )


add_bg_from_local("img/data.jpg")

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
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("css/pica.css")

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
application = get_wsgi_application()

from django.contrib.auth import authenticate

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        user = authenticate(
            username=st.session_state["username"], password=st.session_state["password"]
        )

        if user is not None:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        # else:
        #     st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True
    
    



if check_password():
    switch_page('twitter')
# st.title('Home')