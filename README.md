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

The current dataset has 815 records from Quikr (India, up to ~2019). Adding global data from 2020–2026 will dramatically improve both coverage and accuracy.

### Why More Data Helps
- More records → model generalizes better
- Newer years → the model can price 2020–2026 vehicles including EVs and Hybrids
- Global markets → covers Toyota (USA), BMW (Europe), Kia (Korea), etc.
- More features → engine CC, BHP, transmission, number of owners push R² well above 0.80

---

## 🌍 Where to Find More Data — Worldwide (2020–2026)

> When you share new data, just make sure it has these columns (or equivalent):
> `name, company, year, kms_driven, fuel_type, Price` — add `country` for global models.

---

### 🇺🇸 United States

| Source | Link | Notes |
|--------|------|-------|
| **Kaggle — Used Car Auctions (USA)** | kaggle.com/datasets/tunguz/used-car-auction-prices | 500K+ auction records, 1982–2015, price in USD |
| **Kaggle — Vehicle Sales Data** | kaggle.com/datasets/syedanwarafridi/vehicle-sales-data | USA market, 2014–2015, 550K rows |
| **Kaggle — USA Used Cars (CarGurus)** | kaggle.com/datasets/ananaymital/us-used-cars-dataset | 3M+ listings scraped from CarGurus |
| **CarGurus** | cargurus.com | Live USA listings — price, mileage, year, trim |
| **AutoTrader USA** | autotrader.com | Large volume, includes Carfax history |
| **Cars.com** | cars.com | Covers all 50 states, rich filter options |
| **Craigslist (via UCSD dataset)** | kaggle.com/datasets/austinreese/craigslist-carstrucks-data | 420K Craigslist listings, updated to 2021 |

---

### 🇬🇧 United Kingdom

| Source | Link | Notes |
|--------|------|-------|
| **Kaggle — UK Used Car Dataset** | kaggle.com/datasets/adityadesai13/used-car-dataset-100000-datapoints | 100K UK cars: Audi, BMW, Ford, Toyota, etc. |
| **AutoTrader UK** | autotrader.co.uk | UK's largest used car marketplace |
| **Motors.co.uk** | motors.co.uk | Real listings with mileage, fuel, transmission |
| **Gumtree Cars** | gumtree.com/cars | Classifieds — private seller prices |

---

### 🇩🇪 Germany & Europe

| Source | Link | Notes |
|--------|------|-------|
| **Kaggle — AutoScout24 (Europe)** | kaggle.com/datasets/ander289386/cars-dataset | 16K European listings — price in EUR |
| **Kaggle — Used Cars Germany** | kaggle.com/datasets/wspirat/car-sales-data | German market, includes engine size & power |
| **AutoScout24** | autoscout24.com | Pan-European: Germany, Italy, France, Spain, NL |
| **Mobile.de** | mobile.de | Germany's largest car classifieds, very detailed specs |
| **LaCentrale** | lacentrale.fr | France — includes côte Argus official valuations |

---

### 🇦🇪 Middle East

| Source | Link | Notes |
|--------|------|-------|
| **Dubizzle / Bayut** | dubizzle.com/motors | UAE, Saudi, Egypt — prices in AED |
| **YallaMotor** | yallamotor.com | Covers UAE, KSA, Kuwait, Qatar, Bahrain |
| **OpenSooq** | opensooq.com | Pan-Arab classifieds — Arabic & English |
| **Hatla2ee** | hatla2ee.com | Egypt and Gulf used car market |

---

### 🇦🇺 Australia & 🇨🇦 Canada

| Source | Link | Notes |
|--------|------|-------|
| **CarsGuide (AU)** | carsguide.com.au | Australia's top used car platform |
| **Drive.com.au** | drive.com.au | Includes ANCAP safety ratings |
| **AutoTrader Canada** | autotrader.ca | Canada-wide, prices in CAD |
| **Kaggle — Canadian Used Cars** | kaggle.com/datasets/rupindersinghrana/used-car-prices-in-canada | Canadian market dataset |

---

### 🌐 Global / Multi-Country Kaggle Datasets (Best Starting Points)

| Dataset | Records | Years | Link |
|---------|---------|-------|------|
| **Used Cars Price Prediction** | 19,000+ | 2000–2023 | kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars |
| **Car Prices Dataset (Global)** | 11,000+ | 1990–2023 | kaggle.com/datasets/sidharth178/car-prices-dataset |
| **Vehicle Dataset (Multi-country)** | 50,000+ | 2010–2024 | kaggle.com/datasets/meruvulikitha/vehicle-dataset |
| **Worldwide Used Cars** | 100K+ | 2000–2022 | kaggle.com/datasets/lepchenkov/usedcarscatalog |

---

### Additional Features to Add (Global Data Unlocks These)

| Feature | Impact on R² | Notes |
|---------|-------------|-------|
| Engine Capacity (CC) | Very High | Bigger engine = higher price |
| Horsepower (BHP/HP) | Very High | Performance cars command premium |
| Transmission (Auto/Manual/CVT) | High | Automatics more expensive in most markets |
| Number of Owners | High | 1st-owner cars are worth 15–25% more |
| Country / Market | High | US prices ≠ UK prices ≠ Indian prices |
| Condition (Excellent/Good/Fair) | High | Certified pre-owned vs private seller |
| EV / Hybrid / Plug-in Hybrid | High | EVs depreciate differently — need separate logic |
| Color | Medium | White, silver, black hold value better |
| City / Region | Medium | NYC vs rural, London vs Manchester |
| Accident History | Medium | Clean title vs salvage title |
| Trim Level | Medium | Base vs Sport vs Luxury variant |

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
