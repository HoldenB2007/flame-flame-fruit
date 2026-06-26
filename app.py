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
        return pd.read_csv('demo_data.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

#Stop everything if the model or data didnt load in correctly
if model is None or data is None:
    st.error("Model or Data could not be loaded.")
    st.stop()  


#Allow user to select a date to see fire risk
st.sidebar.header("Date Selection")
available_dates = data['date'].unique()
selected_date = st.sidebar.selectbox("Select Date", available_dates)
day_df = data[data['date'] == selected_date].copy()

# -- Data Preprocessing --

#Drop the same columns that were dropped during model training
col_drop = ['fire','system:index','date', '.geo', 'T21_max', 'T21_mean', 'T21_stdDev']
X_test = day_df.drop(columns=col_drop, errors='ignore')

#Grab the models predictions for the selected date and add them to the dataframe as a new column
probabilities = model.predict_proba(X_test)
day_df['danger_score'] = probabilities[:, 1] * 100

# -- Map Visualization --
st.subheader(f"Fire Risk Map for {selected_date}")
#center the map around SWCO
topo_map = folium.Map(location=[38.0, -107.5], zoom_start=8, tiles='OpenTopoMap')
def get_color(score):
    if score < 40:
        return 'green'
    elif score < 60:
        return 'yellow'
    elif score < 80:
        return 'orange'
    elif score < 90:
        return 'red'
    else:
        return 'darkred'

#Loop through each row in the df and make the appropriate rectangle on the map
for index, row in day_df.iterrows():
    bounds = [[row['lat'], row['lon']], [row['lat'] + lat_step, row['lon'] + lon_step]]
    color = get_color(row['danger_score'])
    opacity = 0.3
    folium.Rectangle(
        bounds=bounds, 
        color=color, 
        weight=1,
        fill=True, 
        fill_color=color,
        fill_opacity=opacity
    ).add_to(topo_map)

    #if there was actually a fire that day, add a marker to the map
    if row['fire'] == 1:
        folium.Marker(
            location=[row['lat'] + lat_step/2, row['lon'] + lon_step/2],
            icon=folium.Icon(color='red', icon='fire', prefix='fa'),
            popup=f"Fire"
        ).add_to(topo_map)
folium_static(topo_map, width=1200, height=700)
