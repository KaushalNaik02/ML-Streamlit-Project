import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
           'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)

# Step 2: Explore the dataset
print(data.head(20))

print("\nDataset Information:")
print(data.info())

#Gives Max,Mean,Min Values
print("\nBasic Statistical Details:")
print(data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())


# Step 3: Preprocess the Data
X = data.drop('Outcome', axis=1)  
y = data['Outcome']  

print(X)
print(y)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy',accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plotting the Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')  # annot=show values ,fmtd= convert value to numrical, cmap=colors
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
# plt.show()

# Classification Report
cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)

# /////// Streamlit Code Starts From Here/////////
st.markdown(
    """
    <h1 style='color:white; text-align: center;'>🧑‍⚕️ Diabetes Predictor 🧑‍⚕️</h1>
    """,
    unsafe_allow_html=True
)
st.markdown('<h3> </h3>', unsafe_allow_html=True)
# st.markdown('<h3> </h3>', unsafe_allow_html=True)
# st.markdown("""<h3 style='text-align:center;'>Enter Your Age:</h3>""",unsafe_allow_html=True)
Age=st.text_input(' Enter Your Age:',30)

Glucose=st.slider('Select Your Glucose Level:',0,199,100)

BloodPressure=st.slider('Select Your BloodPressure Level: ',0,140,70)

SkinThickness=st.slider('Select Your Skin Thickness(MM): ',0,99,20)

Insulin=st.slider(' Select Your Insulin Level:',0,200,60)

BMI=st.slider(' Select Your BMI Level:',0,100,30)

DiabetesPedigreeFunction=st.slider('Select Your DiabetesPedigreeFunction: ',0.0,1.0,0.2,step=0.1)

Pregnancies= st.text_input('Enter Your Pregnencies:',2)

st.markdown('<h3> </h3>', unsafe_allow_html=True)


if st.button('Detect Diabetes'):
    input_data = pd.DataFrame(
    [[int(Pregnancies),int(Glucose),int(BloodPressure),int(SkinThickness),int(Insulin),int(BMI),float(DiabetesPedigreeFunction),int(Age)]],
    columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    )
    deb_pred = model.predict(input_data)
    if (deb_pred==0):  
        st.markdown('<h5>Ooops...He/She Is Suffering From Diabetes🥲 </h5>',unsafe_allow_html=True)
    else:
        st.markdown('<h5>Great.. He/She Healthy😍 </h5>',unsafe_allow_html=True)
