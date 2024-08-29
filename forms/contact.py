import streamlit as st
import re

def validate_email(email):
    # Regular expression to validate email
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if re.match(email_regex, email):
        return False
    else:
        return True


def contact_form():
    with st.form("contact--form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email-ID")
        msg = st.text_area("What Do You Want To Predict")
        submit_button = st.form_submit_button("submit")

        if submit_button:
            # st.success("Message Sent Successfully🎉")
                if not name:
                    st.error("Please Enter Your Name.")
                elif not email:
                    st.error("Please Enter The Email")
                elif validate_email(email):
                    st.error("Invalid email ID. Please Enter A Correct Email.")
                elif not msg:
                    st.error("Please Enter The Description")
                else:
                    st.success("Success..I Will Contact You Soon 🤖")
