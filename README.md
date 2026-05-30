# 🚕 NYC Ride Duration Predictor

Predicts Uber/Lyft trip duration in minutes using real NYC TLC data (January 2026) — trained on 200K trips sampled from 20M+ records.

---

## 📌 Problem Statement

Given trip details available at the time of a ride request — distance, pickup/dropoff zone, time of day, and ride type — predict how long the trip will take in minutes.

This mirrors real-world ETA systems used by Uber and Lyft, built using publicly available government data from the NYC Taxi & Limousine Commission.

---

## 📊 Dataset

- **Source:** [NYC TLC High Volume FHV Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Period:** January 2026
- **Raw size:** 20M+ rows, 25 columns
- **Working sample:** 200,000 rows after cleaning and filtering

---

## 🏗️ Project Structure

```
mlproject/
├── data/                        # Raw and cleaned data (not tracked in git)
├── notebook/
│   ├── EDA.ipynb                # Exploratory data analysis
│   └── Model Training.ipynb     # Model training and evaluation
├── src/
│   ├── components/              # Data ingestion, transformation
│   ├── pipeline/                # Training and prediction pipeline
│   ├── exception.py             # Custom exception handling
│   ├── logger.py                # Logging setup
│   └── utils.py                 # Utility functions
├── models/                      # Saved model artifacts (not tracked in git)
├── requirements.txt
└── README.md
```

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `trip_miles` | Distance of the trip |
| `hour` | Hour of ride request (0–23) |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | 1 if Saturday or Sunday |
| `wait_time` | Seconds between request and pickup |
| `is_uber` | 1 for Uber, 0 for Lyft |
| `is_airport` | 1 if airport fee was charged |
| `PULocationID` | Pickup taxi zone |
| `DOLocationID` | Dropoff taxi zone |
| `congestion_surcharge` | Proxy for traffic/congestion |
| `shared_request_flag` | 1 if shared ride requested |

---

## 🤖 Model Comparison

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 5.35 mins | 7.65 mins | 0.6472 |
| Decision Tree | 5.72 mins | 8.65 mins | 0.5493 |
| Random Forest | 4.06 mins | 6.09 mins | 0.7761 |
| LightGBM | 3.83 mins | 5.80 mins | 0.7972 |
| **XGBoost** ✅ | **3.70 mins** | **5.65 mins** | **0.8075** |

XGBoost selected as final model with best performance across all metrics.

---

## 🛠️ Tech Stack

- **Data Processing:** Python, Pandas, NumPy
- **Modeling:** Scikit-learn, XGBoost, LightGBM

---

## 📁 Setup

```bash
git clone https://github.com/sanvisharma29/mlproject.git
cd mlproject
pip install -r requirements.txt
```

Download the dataset from [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and place it in the `data/` folder.

---

## 📈 Results

XGBoost achieved an MAE of **3.70 minutes** and R² of **0.80** on 200K real NYC Uber/Lyft trips — meaning predictions are within ~4 minutes of actual trip duration on average.

---

## 👤 Author

**SANVI SHARMA**

[![GitHub](https://img.shields.io/badge/GitHub-000?style=flat&logo=github)](https://github.com/sanvisharma29)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/sanvisharma29)