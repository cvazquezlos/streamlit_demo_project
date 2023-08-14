import streamlit as st
from pickle import load

with open("../models/logistic_default.sav", 'rb') as f:
    model = load(f)

st.title("Titanic - Model prediction")

Pclass = st.slider("Pclass", min_value = 0.0, max_value = 3.0, step = 0.5)
Fare = st.slider("Fare", min_value = 0.0, max_value = 3.0, step = 0.5)
Sex_n = st.slider("Sex_n", min_value = 0.0, max_value = 1.0, step = 1.0)
Embarked_n = st.slider("Embarked_n", min_value = 0.0, max_value = 3.0, step = 0.5)
FamMembers = st.slider("FamMembers", min_value = 0.0, max_value = 3.0, step = 0.5)

if st.button("Predict"):
    data_a_predecir = [[Pclass, Fare, Sex_n, Embarked_n, FamMembers]]
    prediction = model.predict(data_a_predecir)[0]
    st.write("Prediction", prediction)