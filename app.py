import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
import torch
import torch.nn as nn
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

#The ADAM model references this class, so I just manually defined it here
class FireNet(nn.Module):
    def __init__(self, input_dim):
        super(FireNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # outputs a probability between 0 and 1
        )

    def forward(self, x):
        return self.network(x).squeeze(1)



#load the model in from directory and cache it
#We only need to do this for the random forest model
@st.cache_resource
def load_model():
    try:
        return joblib.load('fire_prediction_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
rf_model = load_model()

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
if rf_model is None or data is None:
    st.error("Model or Data could not be loaded.")
    st.stop()  


#Allow user to select a date to see fire risk
st.sidebar.header("Date Selection")
available_dates = data['date'].unique()
selected_date = st.sidebar.selectbox("Select Date", available_dates)
day_df = data[data['date'] == selected_date].copy()

# -- Data Preprocessing --

#draw the grid using the gid_x and grid_y columns
#grab all the corners
min_x = day_df['grid_x'].min()
max_x = day_df['grid_x'].max()
min_y = day_df['grid_y'].min()
max_y = day_df['grid_y'].max()

#scale the grid to lat/lon coordinates for the map
day_df['lon'] = np.interp(day_df['grid_x'], [min_x, max_x], [-109.0, -102.0])
day_df['lat'] = np.interp(day_df['grid_y'], [min_y, max_y], [41.0, 37.0])

#Drop the same columns that were dropped during model training
col_drop = ['fire','system:index','date', '.geo', 'T21_max', 'T21_mean', 'T21_stdDev','lat','lon']
X_test = day_df.drop(columns=col_drop, errors='ignore')

#If there are any missing values, fill them with 0, so the square isnt displayed
X_test = X_test.fillna(0)

#Grab the models predictions for the selected date and add them to the dataframe as a new column
if model_choice == "Random Forest":
    probabilities = rf_model.predict_proba(X_test)
    day_df['danger_score'] = probabilities[:, 1] * 100
else: 
    #Load the scaler and scale the data so its the same as the training data for the ADAM model
    scaler = joblib.load('pytorch_scaler.pkl')
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_test_scaled.shape[1] #input dimension for the model
    pytorch_model = FireNet(input_dim) #empty FireNet shell model

    #Load the trained model weights into the FireNet model
    pytorch_model.load_state_dict(torch.load('pytorch_fire_model.pth', weights_only=True))
    pytorch_model.eval()

    #Run predictions using the scaled data, then add it as a danger score same as the random forest model
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        probs = pytorch_model(X_test_tensor).cpu().numpy()
        day_df['danger_score'] = probs * 100


# -- Map Visualization --
st.subheader(f"Fire Risk Map for {selected_date}")

#center the map around SWCO
topo_map = folium.Map(location=[38.0, -107.5], zoom_start=8, tiles='OpenTopoMap')
#Color in the grid based on the models confidence of fire risk
def get_color(score):
    if pd.isna(score):
        return 'transparent'
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

#calculate size of grid squares
#This gives about 6x6 mile squares, which is the size of the grid used in the model training
lon_step = 7.0 / max(1, (max_x - min_x))
lat_step = 4.0 / max(1, (max_y - min_y))

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
#draw the map
folium_static(topo_map, width=1200, height=700)
