import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

ogdata = pd.read_csv('house.csv',comment='#')

data = pd.read_csv('house.csv',comment='#')
data['SEX'].replace(['Male','Female'],[0,1],inplace=True)
data['JOB'].replace(['Government','Private'],[0,1],inplace=True)


x = data.drop(columns=['HOUSE_BUY'])
y = data['HOUSE_BUY']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')



# /////////////streamlit code here/////////////////
st.markdown(
    """
    <h1 style='color:white; text-align: center;'>🏠 HOUSE BUY PREDICTOR 🏠</h1>
    """,
    unsafe_allow_html=True
)
   
data = pd.read_csv('house.csv',comment='#')
st.markdown('<h3> </h3>', unsafe_allow_html=True)

SEX = st.selectbox('Select Your Gender : ',data['SEX'].unique())
JOB = st.radio('Select Your Job :', data['JOB'].unique())
SALARY= st.slider('Your Salary  :',0,99000)
LOAN= st.text_input('Your Loan Amount:',200000)

if st.button('Predict'):
    input_data = pd.DataFrame(
    [[SEX,JOB,SALARY,LOAN]],
    columns=['SEX','JOB','SALARY','LOAN']
)
    
    st.write(input_data)
    input_data['SEX'].replace(['Male','Female'],[0,1],inplace=True)
    input_data['JOB'].replace(['Government','Private'],[0,1],inplace=True)

    house_pred = model.predict(input_data)
    if (house_pred==0):  
        # st.subheader("Ooops...He/She Cant Able TO Buy the House Due To Financial Situation 🥲")
        st.markdown('<h5>Ooops...He/She Cant Able TO Buy the House Due To Financial Situation 🥲 </h5>',unsafe_allow_html=True)
    else:
        st.markdown('<h5>Great.. He/She Can Buy The Dream House 😍 </h5>',unsafe_allow_html=True)