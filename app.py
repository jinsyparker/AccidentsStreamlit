import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
from datetime import datetime
import polyline
import folium
from streamlit_folium import folium_static

# Function to load preprocessed data
@st.cache_data
def load_data():
    return pd.read_csv('accidents_updated.csv')

accidents = load_data()

# Split Data
X = accidents.drop('Accident_Severity', axis=1)
y = accidents['Accident_Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

model_lr = LogisticRegression()
accuracy_lr, precision_lr, recall_lr, f1_lr = train_and_evaluate(model_lr, X_train_scaled, y_train, X_test_scaled, y_test)

# Display Model Performance
st.title("Accident Severity Prediction")

st.subheader("Model Performance")
st.write(f'Logistic Regression - Accuracy: {accuracy_lr:.2f}, Precision: {precision_lr:.2f}, Recall: {recall_lr:.2f}, F1 Score: {f1_lr:.2f}')

# Function to get weather data
def get_weather(api_key, latitude, longitude):
    endpoint = "https://api.openweathermap.org/data/3.0/onecall"
    params = {'lat': latitude, 'lon': longitude, 'units': 'imperial', 'appid': api_key}
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'current' in data and 'weather' in data['current'] and len(data['current']['weather']) > 0:
            current_weather_code = data['current']['weather'][0]['id']
            current_wind_speed = data['current']['wind_speed']
            weather_category = classify_weather_by_code(current_weather_code)
            high_winds = classify_wind_speed(current_wind_speed)
            return weather_category, high_winds
        else:
            st.error("Unexpected weather data format")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None, None

def classify_weather_by_code(weather_code):
    weather_code = int(weather_code)
    if 200 <= weather_code < 600:
        return 4  # Raining
    elif 600 <= weather_code < 700:
        return 5  # Snowing
    elif weather_code == 741 or weather_code == 701:
        return 1  # Fine
    elif weather_code == 741:
        return 2  # Fog or mist
    elif weather_code >= 800:
        return 1  # Fine
    else:
        return 3  # Other

def classify_wind_speed(wind_speed):
    high_winds = 1 if wind_speed >= 45 else 0
    return high_winds

def get_current_day_label():
    current_day = datetime.now().strftime('%A')
    day_mapping = {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6}
    return day_mapping.get(current_day)

def get_current_hour_label():
    current_time = datetime.now().time()
    current_hour = current_time.hour + current_time.minute / 60  
    hour_mapping = {(15.0, 18.59): 0, (19.0, 22.59): 1, (5.0, 9.59): 2, (23.0, 23.59): 3, (0.0, 4.59): 3,(10.0,14.99):4}
    for (start_hour, end_hour), label in hour_mapping.items():
        if start_hour <= current_hour <= end_hour:
            return label
    return 3

# Prediction
st.subheader("Predict Accident Severity")

origin = st.text_input("Enter the origin city:", "Birmingham, UK")
destination = st.text_input("Enter the destination city:", "Manchester, UK")
if st.button("Predict"):
    api_key = 'AIzaSyD-YGrU-YoO6EjBsOw6bwOFMPlHKUkC8vk'
    ow_api_key = '947ad95abdfdbd4a532f992a49f5763b'
    mode = 'driving'
    url = f'https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&mode={mode}&key={api_key}&region=uk&alternatives=true'
    response = requests.get(url)
    
    if response.status_code == 200:
        api_response = response.json()
        if 'routes' in api_response and len(api_response['routes']) > 0:
            routes = api_response['routes'][:3]  # Get only the first 3 routes
            # Create a map centered around the origin
            map_center = api_response['routes'][0]['legs'][0]['start_location']
            m = folium.Map(location=[map_center['lat'], map_center['lng']], zoom_start=10)
            
            color_map = {0: 'red', 1: 'orange', 2: 'yellow'}

            for route_number, route in enumerate(routes, 1):
                legs_info = route['legs']
                st.write(f"Route {route_number}:")
                for leg_number, leg in enumerate(legs_info, 1):
                    st.write(f"Leg {leg_number}:")
                    st.write(f"   Start Address: {leg['start_address']}")
                    st.write(f"   End Address: {leg['end_address']}")
                    st.write(f"   Total Distance: {leg['distance']['text']}")
                    st.write(f"   Total Duration: {leg['duration']['text']}")
                    
                    for step_number, step in enumerate(leg['steps'], 1):
                        st.write(f"Step {step_number}:")
                        st.write(f"   Distance: {step['distance']['text']}")
                        st.write(f"   Duration: {step['duration']['text']}")
                        st.write(f"   Start Location: {step['start_location']}")
                        st.write(f"   End Location: {step['end_location']}")
                        st.write(f"   Instructions: {step['html_instructions']}")
                        polyline_points = step['polyline']['points']
                        coordinates = polyline.decode(polyline_points)
                        avg_latitude = sum(coord[0] for coord in coordinates) / len(coordinates)
                        avg_longitude = sum(coord[1] for coord in coordinates) / len(coordinates)
                        
                        weather_category, high_winds = get_weather(ow_api_key, avg_latitude, avg_longitude)
                        current_day_label = get_current_day_label()
                        current_hour_label = get_current_hour_label()
                        
                        pred = np.array([[current_day_label, weather_category, avg_latitude, avg_longitude, current_hour_label, high_winds]])
                        new_pred_scaled = scaler.transform(pred)
                        
                        prediction_lr = model_lr.predict(new_pred_scaled)[0]
                        probabilities_lr = model_lr.predict_proba(new_pred_scaled)[0]
                        
                        if prediction_lr == 0:
                            severity = 'Fatal'
                        elif prediction_lr == 1:
                            severity = 'Severe'
                        else:
                            severity = 'Slight'
                        
                        st.write(f'Logistic Regression Prediction: {prediction_lr} ({severity})')
                        st.write(f'Logistic Regression Probabilities:')
                        st.write(f'   Fatal: {probabilities_lr[0]:.4f}')
                        st.write(f'   Severe: {probabilities_lr[1]:.4f}')
                        st.write(f'   Slight: {probabilities_lr[2]:.4f}')
                        
                        color = color_map.get(prediction_lr, 'blue')
                        folium.PolyLine(coordinates, color=color, weight=5, opacity=0.7).add_to(m)

            folium_static(m)
        else:
            st.error("No routes found.")
    else:
        st.error(f"Error: {response.status_code}, {response.text}")
