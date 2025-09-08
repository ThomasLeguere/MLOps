import streamlit as st
import joblib

regression_model = joblib.load("regression.joblib")
size = st.number_input("size", 1, None, 1, 1)
nb_rooms = st.number_input("nb_rooms", 1, None, 1, 1)
garden = st.number_input("garden", 0, 1, 0, 1)

result = regression_model.predict([[size, nb_rooms, garden]])

st.write(result)
