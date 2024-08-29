import streamlit as st

about = st.Page(
    page="views/about.py",
    title="About-us",
    icon="🧑‍💼",
    default=True,
    )

car_pred = st.Page( 
    page="views/car_pred.py",
    title="Car-prediction",
    icon="🚘",
)

house_pred = st.Page( 
    page="views/house_pred.py",
    title="House-prediction",
    icon="🏠",
)

iris_pred = st.Page( 
    page="views/iris_pred.py",
    title="Iris-prediction",
    icon="💮",
)

diabetes_pred = st.Page( 
    page="views/diabetes_pred.py",
    title="Diabetes-prediction",
    icon="🧑‍⚕️",
)

# pg = st.navigation(pages=[about,car_pred])
pg = st.navigation(
    {
    "Its-me Robo" : [about],
    "Car" : [car_pred],
    "House" : [house_pred],
    "Iris-flower" : [iris_pred],
    "Diabetes" : [diabetes_pred]

    }
)

st.logo("assets/PredictionLogoFinalBlue.png")
st.sidebar.text("Made With Streamlit💓")

pg.run()