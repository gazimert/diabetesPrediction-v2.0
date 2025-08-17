from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Model ve scaler yükleniyor
model = load_model('mlp_model.h5')
scaler = joblib.load('scaler.pkl')

# Eğitimde kullanılan sütunlar (sıralı şekilde)
EXPECTED_COLUMNS = [
    'age',
    'hypertension',
    'heart_disease',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level',
    'gender_Female',
    'gender_Male',
    'gender_Other',
    'smoking_history_No Info',
    'smoking_history_current',
    'smoking_history_ever',
    'smoking_history_former',
    'smoking_history_never',
    'smoking_history_not current'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("Gelen veri:", data)

        # Gelen veriyi DataFrame'e çevir
        df = pd.DataFrame([data])

        # Eksik olan sütunları 0 ile doldur
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # Fazlalıkları at ve sıralamayı sabitle
        df = df[EXPECTED_COLUMNS]

        # Ölçekleme
        scaled = scaler.transform(df)

        # Tahmin
        prediction_prob = model.predict(scaled)[0][0]
        prediction = int(prediction_prob > 0.5)

        return jsonify({
            "prediction": prediction,
            "probability": float(prediction_prob)
        })

    except Exception as e:
        print("Hata:", str(e))
        return jsonify({"error": "Geçersiz veri", "details": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
