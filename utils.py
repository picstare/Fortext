import streamlit as st


def logout():
    del st.session_state["password_correct"]