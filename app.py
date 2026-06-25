import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static, st_folium

# -- Basic UI --
st.set_page_config(page_title="Fire Prediction Demo", page_icon="🔥", layout="wide")
st.title("Fire Prediction Demo")
st.write("This is a demo application for predicting wildfires")

#Prediction Model Selection
st.sidebar.header("ML Engine")
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Descent"])
st.sidebar.divider()

# -- Model Specfics --

#Make sure the model names match the filenames saved in directory
model_file='fire_prediction_model.pkl' if model_choice == "Random Forest" else 'fire_prediction_model_gd.pkl'

#load the model in from directory and cache it
@st.cache_resource
def load_model(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
model = load_model(model_file)

# -- Map data --
@st.cache_data
def load_data():
    try:
        return pd.read_csv('final_fire_data.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

#Stop everything if the model or data didnt load in correctly
if model is None or data is None:
    st.error("Model or Data could not be loaded.")
    st.stop()  