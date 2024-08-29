
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

data = pd.read_csv('car.csv')

num_data =  pd.read_csv('car.csv')
num_data['name'].replace(['santro','ertiga','swift','omni','audi','xuv'],[1,2,3,4,5,6],inplace=True)
num_data['fuel'].replace(['petrol','diesel','cng'],[1,2,3],inplace=True)
print(num_data)

input_data = num_data.drop(columns=['selling_price'])
output_data = num_data['selling_price']


(x_train,x_test,y_train,y_test)=train_test_split(input_data,output_data,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)




# //////////////Streamlit code starts here/////////////////////
num_data =  pd.read_csv('car.csv') 
st.markdown(
    """
    <h1 style='color:white; text-align: center;'>🚘 CAR  PRICE  PREDICTOR 🚘</h1>
    """,
    unsafe_allow_html=True
)
st.header('')
name=st.selectbox('Select Car Brand',num_data['name'].unique())
fuel=st.selectbox('Select Car Fuel-Type',num_data['fuel'].unique())
average=st.slider('Approximate Mileage Of Car',10,30,15)
launch=st.slider('Launching Year Of Car',2004,2024,2015)

seats=st.slider('Number of Seats in Car',5,9,6)
horsepower=st.slider('Horsepower Of Car',100,900,500)

if st.button('Predict'):
    input_data = pd.DataFrame(
    [[name,fuel,average,launch,seats,horsepower]],
    columns=['name','fuel','average','launch','seats','horsepower']
)
    
    input_data['name'].replace(['santro','ertiga','swift','omni','audi','xuv'],[1,2,3,4,5,6],inplace=True)
    input_data['fuel'].replace(['petrol','diesel','cng'],[1,2,3],inplace=True)

    st.write(input_data)

    car_price = model.predict(input_data)
    st.subheader('Car Price Is going To Be : '+str(car_price[0])) 