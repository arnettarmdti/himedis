import xgboost as xgb
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db
import json
import time

# Memuat model XGBoost yang sudah dilatih
model = joblib.load('xgboost_model3.pkl')

# Mengonfigurasi Firebase
firebase_creds = {
    "type": st.secrets["firebase"]["type"],
    "project_id": st.secrets["firebase"]["project_id"],
    "private_key_id": st.secrets["firebase"]["private_key_id"],
    "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
    "client_email": st.secrets["firebase"]["client_email"],
    "client_id": st.secrets["firebase"]["client_id"],
    "auth_uri": st.secrets["firebase"]["auth_uri"],
    "token_uri": st.secrets["firebase"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
}

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://himedis-default-rtdb.firebaseio.com/'
    })

# Mengakses Realtime Database
ref = db.reference('/dataSensor')

# Fungsi prediksi
def predict(sensor_value_ir, sensor_value_red):
    features = np.array([sensor_value_ir, sensor_value_red]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return float(prediction)

# Fungsi untuk membaca data dari Firebase, memprosesnya, dan mengirimkan hasilnya
def process_data():
    while True:
        # Membaca data terbaru dari Firebase
        snapshot = ref.order_by_key().limit_to_last(1).get()
        if snapshot:
            for key, data in snapshot.items():
                sensor_value_ir = data.get('sensor_value_ir')
                sensor_value_red = data.get('sensor_value_red')

                if sensor_value_ir is not None and sensor_value_red is not None:
                    prediction = predict(sensor_value_ir, sensor_value_red)
                    result = {
                        'sensor_value_ir': sensor_value_ir,
                        'sensor_value_red': sensor_value_red,
                        'prediction': prediction
                    }
                    # Mengirimkan hasil prediksi ke Firebase
                    ref.child(key).update({'prediction': prediction})
                    print(f"Processed data: {result}")
                else:
                    print("Invalid data format")

        # Tunggu sebelum memeriksa data lagi
        time.sleep(10)

if __name__ == "__main__":
    process_data()
