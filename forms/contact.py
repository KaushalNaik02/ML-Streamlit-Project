import streamlit as st
import re
import pandas as pd
import os

def validate_email(email):
    # Regular expression to validate email
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if re.match(email_regex, email):
        return False
    else:
        return True
    


def save_to_csv(name, email, msg, file_name="form_data.csv"):
    file_path = os.path.abspath(file_name)

    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["Name", "Email", "Message"])
        df.to_csv(file_path, index=False)

    new_data = pd.DataFrame({"Name": [name], "Email": [email], "Message": [msg]})
    new_data.to_csv(file_path, mode="a", header=False, index=False)
    # st.write(f"Data saved to: {file_path}")

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
                    save_to_csv(name, email, msg)
                    st.success("Success..I Will Contact You Soon 🤖")
                














