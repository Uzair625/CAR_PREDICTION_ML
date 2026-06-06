# 🚗 Car Price Prediction ML

A machine learning web app that predicts the **resale price of used cars** based on brand, model, year, fuel type, and kilometers driven — built with scikit-learn and deployed with Streamlit.

---

## 🔗 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

> Deploy yourself: see the **Deployment** section below.

---

## 📌 Project Overview

Used car prices depend on many factors — age, brand, mileage, and fuel type. This project trains several ML regression models on real used-car transaction data to predict a fair resale price for any given vehicle.

### Problem Type
**Supervised Regression** — predicting a continuous numerical target (price in Indian Rupees ₹).

---

## 📂 Repository Structure

```
CAR_PREDICTION_ML/
├── app.py               # Streamlit web application
├── train_model.py       # Model training script (run this to regenerate model.pkl)
├── cleaned_car.csv      # Cleaned dataset (815 used-car records)
├── model.pkl            # Trained sklearn Pipeline (preprocessor + best model)
├── model_results.pkl    # Evaluation scores for all 3 trained models
├── requirements.txt     # Python dependencies
└── .gitignore
```

---

## 🗂️ Dataset

**File:** `cleaned_car.csv`  
**Source:** Quikr used-car listings (India)  
**Records:** 815 used-car transactions

| Column | Type | Description |
|--------|------|-------------|
| `name` | Categorical | Full car model name (e.g., "Hyundai Grand i10") |
| `company` | Categorical | Brand / manufacturer (e.g., Hyundai, Maruti, Ford) |
| `year` | Numerical | Model year (2003–2019) |
| `kms_driven` | Numerical | Total kilometers driven |
| `fuel_type` | Categorical | Petrol / Diesel / LPG |
| `Price` | Numerical | **Target** — resale price in ₹ |

---

## ⚙️ ML Pipeline

### 1. Feature Engineering
- **One-Hot Encoding** (`pd.get_dummies` / `OneHotEncoder`) applied to all categorical columns: `name`, `company`, `fuel_type`
- **Standard Scaling** (`StandardScaler`) applied to numerical columns: `year`, `kms_driven`

### 2. Train / Test Split
- **80% training / 20% testing** using `train_test_split(random_state=42)`

### 3. Models Trained

| Model | R² Score | MAE (₹) |
|-------|----------|---------|
| **Linear Regression** ✅ Best | **0.6424** | **1,14,032** |
| Random Forest Regressor | 0.5413 | 1,23,221 |
| Gradient Boosting Regressor | 0.5009 | 1,55,857 |

The best-performing model (Linear Regression) is saved as a full sklearn **Pipeline** that includes the preprocessor — so no separate preprocessing step is needed at inference time.

### 4. Key Features (Importance)
- **Model Year** — depreciation is the strongest price driver
- **Brand / Company** — premium brands (Audi, BMW) retain value; budget brands drop faster
- **Kms Driven** — higher mileage = lower price
- **Fuel Type** — diesel vehicles often hold value better for high-mileage use cases
- **Car Model** — specific model reputation (e.g., Swift, i10) affects price

---

## 🖥️ Streamlit App

The app (`app.py`) provides:
- **Sidebar inputs:** Brand, Car Model (filtered by brand), Year, Fuel Type, Kms Driven
- **Price prediction card** showing estimated resale value in ₹ and USD
- **Market range** — min/avg/max prices for similar cars from the dataset
- **Depreciation & mileage insights**
- **Model performance panel** showing R² and MAE for all 3 models

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Uzair625/CAR_PREDICTION_ML.git
cd CAR_PREDICTION_ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain the model
python train_model.py

# 4. Launch the app
streamlit run app.py
```

---

## 🚀 Deployment (Streamlit Cloud)

> **Note:** Vercel is for frontend/Node.js apps. Streamlit ML apps deploy on [Streamlit Community Cloud](https://share.streamlit.io) — free for public repos.

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Select repo: `Uzair625/CAR_PREDICTION_ML`, branch: `main`
4. Main file path: `app.py`
5. Click **Deploy** — dependencies auto-install from `requirements.txt`

---

## 📈 How to Improve Accuracy (More & Newer Data)

The current dataset has 815 records from Quikr (India, up to ~2019). Adding more recent data (2020–2026) will significantly improve predictions. See the **Data Sources** section below.

### Why More Data Helps
- More records → model generalizes better
- Newer years → the model can price 2020–2026 vehicles
- More brands/models → better coverage for rare makes

---

## 🔍 Where to Find More Data (2020–2026)

### Free / Open Datasets

| Source | Link | Notes |
|--------|------|-------|
| **Kaggle — Used Car Listings** | [kaggle.com/datasets](https://www.kaggle.com/datasets?search=used+car+price+india) | Search "used car price India 2024" — many updated datasets |
| **Kaggle — CarDekho Dataset** | kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho | Updated versions available with 2020–2023 data |
| **Kaggle — Cars4U Dataset** | kaggle.com/datasets/avikasliwal/used-cars-price-prediction | ~7,000 records, broader year range |

### Live Scraping (Real-Time 2024–2026 Data)

| Platform | URL | What you get |
|----------|-----|-------------|
| **CarDekho** | cardekho.com/used-cars | India's largest used car portal — real listings |
| **CarWale** | carwale.com/used | Rich spec data including engine CC, BHP |
| **OLX Autos** | olx.in/cars | High volume of listings |
| **Quikr Cars** | quikr.com/cars | Original data source for this dataset |
| **Spinny** | spinny.com | Certified pre-owned cars with verified mileage |

> To scrape: use Python `requests` + `BeautifulSoup` or `Selenium`. Always check the site's `robots.txt` and Terms of Service first.

### Additional Features You Can Add (with richer data)

| Feature | Impact |
|---------|--------|
| Engine Capacity (CC) | High — bigger engine = higher price |
| Horsepower (BHP) | High — performance vehicles command premium |
| Number of Owners | Medium — 1st owner car worth more |
| Transmission (Auto/Manual) | Medium — automatics more expensive |
| Color | Low-Medium — white/silver hold value better |
| Insurance Validity | Low |
| City / Location | Medium — metro prices differ from tier-2 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| pandas | Data loading and manipulation |
| scikit-learn | ML models, preprocessing pipeline |
| Streamlit | Web UI |
| pickle | Model serialization |

---

## 👤 Author

**Uzair Ali** — [GitHub @Uzair625](https://github.com/Uzair625)
