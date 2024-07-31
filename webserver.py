import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import json

# Firebase configuration
FIREBASE_URL = 'https://himedis.firebaseio.com/'
FIREBASE_AUTH_KEY = 'baQrBRO8FzJfrWDVyVojeQFzCdjf8xJNpbgrw3y2'  # Ganti dengan kunci auth Firebase Anda

# Load the XGBoost model
model = xgb.XGBRegressor()
model.load_model('xgboost_model3.pkl')  # Sesuaikan jika model Anda berbeda

# Function to make predictions
def predict(ir_value, red_value):
    input_data = np.array([[ir_value, red_value]])
    prediction = model.predict(input_data)
    return prediction[0]

# Function to send data to a new path in Firebase
def send_to_firebase(data, path):
    url = FIREBASE_URL + path + '.json?auth=' + FIREBASE_AUTH_KEY
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        st.success("Data sent to Firebase successfully!")
    else:
        st.error("Failed to send data to Firebase.")

# Function to fetch data from Firebase
def fetch_from_firebase(path):
    url = FIREBASE_URL + path + '.json?auth=' + FIREBASE_AUTH_KEY
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch data from Firebase.")
        return {}

# Streamlit app
def main():
    st.title("Sensor Data Prediction")

    # Fetch data from Firebase
    data = fetch_from_firebase('dataSensor')

    if data:
        # Process each item in the data
        for key, value in data.items():
            ir_value = value.get('irValue', 0)
            red_value = value.get('redValue', 0)
            prediction = predict(ir_value, red_value)

            st.write(f"IR Value: {ir_value}, Red Value: {red_value}")
            st.write(f"Prediction: {prediction}")

            # Send processed data to a new path in Firebase
            result_data = {
                "irValue": ir_value,
                "redValue": red_value,
                "prediction": prediction
            }
            send_to_firebase(result_data, 'processedData/' + key)

if __name__ == "__main__":
    main()
