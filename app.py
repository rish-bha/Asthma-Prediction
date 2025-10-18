
import joblib
import pandas as pd
import streamlit as st
import requests
from streamlit_geolocation import streamlit_geolocation

encoder_path = 'onehot_encoder'
scaler_path = 'scaler'
model_path = 'best_model_RF'

encoder = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

st.title('Asthma Prediction App')
st.write('Enter your details to predict asthma risk:')


# Define options for categorical fields
age_options = ['Above 50', '41-50', '19-30', '31-40']
gender_options = ['Male', 'Female']
outdoor_job_options = ['Occasionally', 'Frequently', 'Rarely']
outdoor_activities_options = ['Extremely likely', 'Neither likely or dislikely',
       'Not at all likely']
smoking_habit_options = ['No', 'Yes']
uvindex_options = ['Low', 'Extreme']
lat=37.5485
lon=-121.9886

city = st.text_input("City", value="Fremont")

def get_coordinates(city_name):
       api_key = "4e514b49d73362c5d739f05fea7f27cd"
       url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
       response = requests.get(url).json()

       if response:
              lat = response[0]['lat']
              lon = response[0]['lon']
              print(f"City: {city}")
              print(f"Latitude: {lat}, Longitude: {lon}")
       else:
              print("City not found!")
       return lat, lon
lat, lon = get_coordinates(city)
# Fetch weather data from OpenWeatherMap API
weather_api_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid=4e514b49d73362c5d739f05fea7f27cd"
weather_data = None
uvi_data = None
uvi_url = f"https://api.openweathermap.org/data/2.5/uvi?lat={lat}&lon={lon}&appid=4e514b49d73362c5d739f05fea7f27cd"

try:
       response = requests.get(weather_api_url)
       if response.status_code == 200:
              weather_data = response.json()
except Exception as e:
       weather_data = None

try:
       response1 = requests.get(uvi_url)
       if response1.status_code == 200:
              uvi_data = response1.json()
except Exception as e:
       uvi_data = None

# Default values
default_humidity = 65
default_pressure = 1009
default_temperature = 29
default_wind_speed = 15

# Pre-fill with weather data if available
if weather_data:
       st.write("Fetched current weather data for Fremont, CA")
       default_humidity = weather_data.get('main', {}).get('humidity', default_humidity)
       default_pressure = weather_data.get('main', {}).get('pressure', default_pressure)
       # Convert Kelvin to Celsius for temperature
       default_temperature = round(weather_data.get('main', {}).get('temp', 302.15) - 273.15)
       default_wind_speed = round(weather_data.get('wind', {}).get('speed', default_wind_speed))

if uvi_data:
       uvi_value = uvi_data.get('value', 0)
       if uvi_value < 5:
              uvindex_default = 0
       else:
              uvindex_default = 1


# Streamlit input fields with pre-filled values
age = st.selectbox('Age', age_options, index=2)
gender = st.selectbox('Gender', gender_options, index=0)
outdoor_job = st.selectbox('Outdoor Job', outdoor_job_options, index=0)
outdoor_activities = st.selectbox('Outdoor Activities', outdoor_activities_options, index=2)
smoking_habit = st.selectbox('Smoking Habit', smoking_habit_options, index=0)
uvindex = st.selectbox('UV Index', uvindex_options, index=uvindex_default)
humidity = st.number_input('Humidity', min_value=0, max_value=100, value=default_humidity)
pressure = st.number_input('Pressure', min_value=900, max_value=1100, value=default_pressure)
temperature = st.number_input('Temperature', min_value=-20, max_value=60, value=default_temperature)
wind_speed = st.number_input('Wind Speed', min_value=0, max_value=100, value=default_wind_speed)


if st.button('Predict'):
       user_data = pd.DataFrame([{
              "Age": age,
              "Gender": gender,
              "OutdoorJob": outdoor_job,
              "OutdoorActivities": outdoor_activities,
              "SmokingHabit": smoking_habit,
              "UVIndex": uvindex,
              "Humidity": humidity,
              "Pressure": pressure,
              "Temperature": temperature,
              "WindSpeed": wind_speed
       }])

       # categorical columns (must match training)
       cat_columns = ['Age', 'Gender', 'OutdoorJob', 'OutdoorActivities', 'SmokingHabit', 'UVIndex']
       num_columns = ['Humidity', 'Pressure', 'Temperature', 'WindSpeed']

       # Transform categorical data
       user_transform = encoder.transform(user_data[cat_columns])
       feature_names = encoder.get_feature_names_out(cat_columns)
       user_transform_df = pd.DataFrame(user_transform, columns=feature_names)

       # Drop categorical cols and combine with numeric
       user_drop = user_data.drop(cat_columns, axis=1).reset_index(drop=True)
       user_encoded = pd.concat([user_drop, user_transform_df], axis=1)

       continuous_columns = [col for col in num_columns]  # same as training
       user_encoded[continuous_columns] = scaler.transform(user_encoded[continuous_columns])

       prediction = model.predict(user_encoded)
       prediction = prediction[0]
       st.subheader('Prediction Result:')
       st.write(f'Predicted Asthma Risk: {prediction}')