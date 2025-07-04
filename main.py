from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf




app = FastAPI()

#model
model = tf.keras.models.load_model("./model_ann_new", compile=False)


#norm data
scaler = joblib.load("./scaler.pkl")

#encode kondisi tandah
le_kondisi = joblib.load("./le_kondisi.pkl")

#encode rekomendasi tanah
le_rekomendasi =joblib.load("./le_rekomendasi.pkl")

#fitur sesuai training urutannya gaboleh diubah
feature_order = [
    'Nitrogen (mg/kg)', 'Fosfor (mg/kg)', 'Kalium (mg/kg)', 'pH', 
    'Kelembapan (%)', 'Suhu (°C)', 'EC (dS/m)', 'Salinitas (ppt)', 'TDS (ppm)'
]

#input
class InputData(BaseModel):
    nitrogen: float
    fosfor: float
    kalium: float
    ph: float
    kelembapan: float
    suhu: float
    ec: float
    salinitas: float
    tds: float

@app.post("/predict")
def predict(data: InputData):
    try:
        # Ambil data dan ubah ke array
        input_data = [
            data.nitrogen, data.fosfor, data.kalium, data.ph, 
            data.kelembapan, data.suhu, data.ec, data.salinitas, data.tds
        ]
        input_array = np.array([input_data])

        # Preprocessing
        input_scaled = scaler.transform(input_array)


        pred_kondisi, pred_rekomendasi = model.predict(input_scaled)

        # Ambil indeks dengan probabilitas tertinggi
        kondisi_class = int(np.argmax(pred_kondisi))
        rekomendasi_class = int(np.argmax(pred_rekomendasi))

    
        label_kondisi = le_kondisi.inverse_transform([kondisi_class])[0]
        label_rekomendasi = le_rekomendasi.inverse_transform([rekomendasi_class])[0]

        return {
            "prediksi_kondisi tanah": label_kondisi,
            "prediksi_rekomendasi tindakan": label_rekomendasi,
            "probabilitas_kondisi tanah": pred_kondisi.tolist(),
            "probabilitas_rekomendasi tindakan": pred_rekomendasi.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
