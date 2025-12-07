# Retail Sales Forecasting Using Machine Learning & Weather Data

A complete end-to-end machine learning system for forecasting daily retail sales using:
- Kaggle retail sales time-series data  
- NOAA weather data (temperature, rainfall)  
- Engineered lagged and rolling features  
- Multiple ML models (Linear Regression, Random Forest, XGBoost)  
- A modern terminal dashboard built with **Rich + Typer**  

---

## 🚀 Features

- **Automated ML Pipeline**
  - Data cleaning  
  - Weather–sales merging  
  - Feature engineering  
  - Model training & evaluation  
  - Metric reports + validation predictions  

- **Multiple ML Models**
  - Linear Regression  
  - Random Forest  
  - XGBoost  

- **Interactive CLI Dashboard**
  - View metrics  
  - Browse predictions  
  - Inspect weather vs sales  
  - Run what-if simulations (change temperature/rainfall)  

- **Professional Project Structure**
  - Clean modular source code  
  - Reproducible pipeline  
  - Metrics & models saved for dashboard use  

---

## 🧱 Project Structure

project/
├── main.py
├── config.py
├── requirements.txt
│
├── dashboard/
│ └── app.py # Rich + Typer Terminal Dashboard
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
├── reports/
│ ├── metrics/
│ └── figures/
│
└── src/
├── data/
├── features/
└── models/


---

## 🔧 Installation

pip install -r requirements.txt


---

## 🏗️ Run the ML Pipeline

Build features, train models, and generate metrics:

python3 main.py


---

## 🖥️ Terminal Dashboard (Rich + Typer)

### 📊 View Model Metrics
python3 dashboard/app.py metrics


### 🔍 Browse Validation Predictions

python3 dashboard/app.py browse

### 🌦️ Inspect Weather vs Sales

python3 dashboard/app.py weather


### 🧪 What-If Simulator
Simulate new temperature & rainfall conditions:


python3 dashboard/app.py simulate



---

## 🧠 Models Trained

- Linear Regression  
- Random Forest  
- XGBoost  

Model artifacts and metric reports are saved automatically after running the pipeline.

---

## 📝 Notes

- This project follows the practicum proposal for **Retail Sales Forecasting Using Machine Learning Techniques**.  
- Weather and sales signals are merged at the daily store level.  
- The terminal dashboard provides fast navigation and clear visualization without requiring a web UI.  

---

## 📌 Author
PallySlakk’s Practicum — Retail Sales Forecasting  
