import pickle
import pandas as pd
import streamlit as st

encoder_path = 'onehot_encoder'
scaler_path = 'scaler'
model_path = 'best_model_RF'
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title('Asthma Prediction App')
st.write('Enter your details to predict asthma risk:')

# Define options for categorical fields
location_options = ['Petaling', 'Kampong Baharu Balakong', 'Putrajaya', 'Kota Bharu',
       'Kuala Lampur', 'Cyberjaya']
age_options = ['Above 50', '41-50', '19-30', '31-40']
gender_options = ['Male', 'Female']
outdoor_job_options = ['Occasionally', 'Frequently', 'Rarely']
outdoor_activities_options = ['Extremely likely', 'Neither likely or dislikely',
       'Not at all likely']
smoking_habit_options = ['No', 'Yes']
uvindex_options = ['Low', 'Extreme']

# Streamlit input fields
location = st.selectbox('Location', location_options)
age = st.selectbox('Age', age_options)
gender = st.selectbox('Gender', gender_options)
outdoor_job = st.selectbox('Outdoor Job', outdoor_job_options)
outdoor_activities = st.selectbox('Outdoor Activities', outdoor_activities_options)
smoking_habit = st.selectbox('Smoking Habit', smoking_habit_options)
uvindex = st.selectbox('UV Index', uvindex_options)
humidity = st.number_input('Humidity', min_value=0, max_value=100, value=65)
pressure = st.number_input('Pressure', min_value=1003, max_value=1020, value=1009)
temperature = st.number_input('Temperature', min_value=-20, max_value=60, value=29)
wind_speed = st.number_input('Wind Speed', min_value=0, max_value=100, value=15)

if st.button('Predict'):
    user_data = pd.DataFrame([{
        "Location": location,
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
    cat_columns = ['Location', 'Age', 'Gender', 'OutdoorJob', 'OutdoorActivities', 'SmokingHabit', 'UVIndex']
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