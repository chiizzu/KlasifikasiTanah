# 🌱 Soil Classification System with Machine Learning and IoT

A smart soil monitoring system that uses **ESP32**, **NPK Sensor**, **machine learning**, and **FastAPI** to classify soil types and provide actionable recommendations for smart farming. This project was built as part of a precision agriculture initiative.

---

## 📌 Overview

This system reads soil data from NPK sensor connected to an ESP32 and classifies the soil type using a trained machine learning model. The result is deployed in local host use FastAPI, showing soil condition and suggested farming actions.

---

## 🧠 Key Features

- 🔍 Classifies soil types based on moisture, temperature, and pH data
- 🤖 Uses machine learning (Decision Tree classifier) for prediction
- 🛰️ IoT integration with ESP32 to send real-time sensor data
- ✅ Displays actionable recommendations (e.g., fertilize, irrigate)

---

## 🗂 Repository Structure

```
│   klasifikasi.ipynb
│   le_kondisi.pkl
│   le_rekomendasi.pkl
│   main.py
│   README.md
│   requirements.txt
│   scaler.pkl
│
├───data
│       create_data.ipynb
│       Data_Sensor_NPK.csv
│
├───model decision tree
│       model_decisiontree.pkl
│
├───model random forest
│       model_random_forest.pkl
│
├───model_ann_new
│   │   fingerprint.pb
│   │   keras_metadata.pb
│   │   saved_model.pb
│   │
│   ├───assets
│   └───variables
│           variables.data-00000-of-00001
│           variables.index
│
└───__pycache__
```

 
---

## 🚀 How to Run the Dashboard

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chiizzu/KlasifikasiTanah.git
   cd KlasifikasiTanah
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
3. **Run the FastAPI**
   ```
   uvicorn main:app --reload

   
---
## 🧪 Sample Dataset
The dataset used for training is embedded in the Jupyter Notebook klasifikasi.ipynb. It includes labeled soil conditions based on various combinations of temperature and moisture.
