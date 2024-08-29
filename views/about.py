import streamlit as st
# st.header("its about us")
from forms.contact import contact_form


@st.dialog("Contact-me")
def show_contact_form():
    contact_form()
   
      


col1 , col2 = st.columns(2,gap="large",vertical_alignment="center")
with col1:
    st.header("")
    st.image("assets/rr1.png")
with col2:
    st.header("")
    st.title("Prediction")
    st.write("Hey I Am The Predictor King. i will help you to predict the some of the things if your things are not listed in the side bar. then please Contact-me and have a patience.")
    if st.button("💌Contact-Me"):
        show_contact_form()