import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Fire Prediction Demo")
st.write("This is a demo application for predicting wildfires")

st.sidebar.header("ML Engine")
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Descent"])
st.sidebar.divider()
model_file='fire_prediction_model.pkl' if model_choice == "Random Forest" else 'fire_prediction_model_gd.pkl'

@st.cache_data
def load_model(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
model = load_model(model_file)

