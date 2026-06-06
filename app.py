import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Global Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem;
}
.sub-header { text-align: center; color: #888; font-size: 1rem; margin-bottom: 1.5rem; }
.price-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 16px; text-align: center; color: white;
    box-shadow: 0 8px 32px rgba(102,126,234,0.4);
}
.price-label { font-size: 0.9rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 2px; }
.price-value { font-size: 2.8rem; font-weight: 800; margin: 0.4rem 0; }
.price-local { font-size: 1.1rem; opacity: 0.85; }
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: 10px;
    padding: 0.7rem 2rem; font-size: 1.05rem; font-weight: 600;
}
.feature-tag {
    display: inline-block; background: #ede9fe; color: #5b21b6;
    border-radius: 20px; padding: 0.2rem 0.7rem; font-size: 0.8rem; margin: 0.15rem;
}
</style>
""", unsafe_allow_html=True)

# ── Exchange rates (USD -> local) ──────────────────────────────────────────────
CURRENCY_BY_COUNTRY = {
    'India':        ('INR', 83.0,   'Rs.'),
    'USA':          ('USD', 1.0,    '$'),
    'Germany':      ('EUR', 0.93,   '€'),
    'France':       ('EUR', 0.93,   '€'),
    'Italy':        ('EUR', 0.93,   '€'),
    'Spain':        ('EUR', 0.93,   '€'),
    'Netherlands':  ('EUR', 0.93,   '€'),
    'Belgium':      ('EUR', 0.93,   '€'),
    'Austria':      ('EUR', 0.93,   '€'),
    'Switzerland':  ('EUR', 0.93,   '€'),
    'UK':           ('GBP', 0.79,   '£'),
    'UAE':          ('AED', 3.67,   'AED'),
    'Pakistan':     ('PKR', 278.0,  'Rs.'),
    'Poland':       ('PLN', 4.02,   'zł'),
    'South Africa': ('ZAR', 18.6,   'R'),
    'Portugal':     ('EUR', 0.93,   '€'),
    'Kenya':        ('KES', 129.0,  'KSh'),
    'Canada':       ('CAD', 1.36,   'CA$'),
    'Australia':    ('AUD', 1.53,   'A$'),
    'Global':       ('USD', 1.0,    '$'),
}

# ── Load model and data ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    # Prefer merged global dataset if available
    global_path = os.path.join('data', 'processed', 'global_cars.csv')
    if os.path.exists(global_path):
        return pd.read_csv(global_path, low_memory=False), True
    df = pd.read_csv('cleaned_car.csv')
    df['country'] = 'India'
    if 'Price' in df.columns:
        df['Price_USD'] = (df['Price'] * 0.012).round(2)
    return df, False

@st.cache_data
def load_results():
    try:
        with open('model_results.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model        = load_model()
car_df, is_global = load_data()
model_results = load_results()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌍 Global Car Price Predictor</div>', unsafe_allow_html=True)
coverage = f"{car_df['country'].nunique()} countries" if is_global else "India"
st.markdown(
    f'<div class="sub-header">ML-powered resale value estimates — {len(car_df):,} listings across {coverage}</div>',
    unsafe_allow_html=True
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Vehicle Details")
    st.markdown("---")

    # Country selector (only if global data available)
    if is_global and 'country' in car_df.columns:
        countries = sorted(car_df['country'].dropna().unique())
        selected_country = st.selectbox("🌍 Market / Country", countries,
                                         index=countries.index('India') if 'India' in countries else 0)
        country_df = car_df[car_df['country'] == selected_country]
    else:
        selected_country = 'India'
        country_df = car_df

    companies = sorted(country_df['company'].dropna().unique())
    default_co = companies.index('Hyundai') if 'Hyundai' in companies else 0
    selected_company = st.selectbox("🏢 Brand / Company", companies, index=default_co)

    model_options = sorted(
        country_df[country_df['company'] == selected_company]['name'].dropna().unique()
    )
    if not model_options:
        model_options = sorted(car_df['name'].dropna().unique())
    selected_model = st.selectbox("🚘 Car Model", model_options)

    years = sorted(car_df['year'].dropna().astype(int).unique(), reverse=True)
    selected_year = st.selectbox("📅 Model Year", years)

    fuel_types = sorted(car_df['fuel_type'].dropna().unique())
    selected_fuel = st.selectbox("⛽ Fuel Type", fuel_types)

    transmission_options = ['Unknown', 'Manual', 'Automatic', 'CVT', 'Semi-Auto']
    if 'transmission' in car_df.columns:
        transmission_options = sorted(car_df['transmission'].dropna().unique())
    selected_transmission = st.selectbox("⚙️ Transmission", transmission_options)

    kms_driven = st.number_input(
        "🛣️ Kilometers Driven", min_value=0, max_value=700000,
        value=30000, step=1000
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)

# ── Main content ───────────────────────────────────────────────────────────────
col_main, col_info = st.columns([2, 1])

with col_main:
    if predict_btn:
        # Build input row — include only columns the model was trained on
        input_dict = {
            'name':         selected_model,
            'company':      selected_company,
            'year':         selected_year,
            'kms_driven':   kms_driven,
            'fuel_type':    selected_fuel,
            'transmission': selected_transmission,
        }
        if is_global:
            input_dict['country'] = selected_country

        input_data = pd.DataFrame([input_dict])

        # Ensure correct column types
        input_data['year']       = int(selected_year)
        input_data['kms_driven'] = float(kms_driven)

        try:
            predicted_usd = float(model.predict(input_data)[0])
            predicted_usd = max(300, predicted_usd)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Convert to local currency
        _, rate, symbol = CURRENCY_BY_COUNTRY.get(selected_country, ('USD', 1.0, '$'))
        predicted_local = predicted_usd * rate

        # Price card
        st.markdown("### 💰 Prediction Result")
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated Resale Value</div>
            <div class="price-value">${predicted_usd:,.0f} USD</div>
            <div class="price-local">{symbol} {predicted_local:,.0f} {
                CURRENCY_BY_COUNTRY.get(selected_country, ('USD',))[0]
            }</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📋 Your Vehicle Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Car Age",     f"{2024 - int(selected_year)} yrs")
        c2.metric("Kms Driven",  f"{kms_driven:,} km")
        c3.metric("Fuel",        selected_fuel)
        c4.metric("Transmission", selected_transmission)

        # Market price range for similar cars
        similar = car_df[
            (car_df['company'] == selected_company) &
            (car_df['year'].between(int(selected_year) - 2, int(selected_year) + 2))
        ]['Price_USD'].dropna()

        if not similar.empty:
            st.markdown("#### 📊 Market Range (Similar Cars — USD)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Min",  f"${similar.min():,.0f}")
            m2.metric("Avg",  f"${similar.mean():,.0f}")
            m3.metric("Max",  f"${similar.max():,.0f}")

        age = 2024 - int(selected_year)
        dep = ("🟢 Low depreciation (new)" if age < 3
               else "🟡 Moderate depreciation" if age < 7
               else "🔴 High depreciation (old)")
        km_tip = ("🟢 Low mileage" if kms_driven < 40000
                  else "🟡 Moderate mileage" if kms_driven < 100000
                  else "🔴 High mileage")
        st.info(f"{dep}   |   {km_tip}")

    else:
        st.markdown("### 👈 Fill in vehicle details on the left")
        st.markdown("""
Select the **market/country**, then choose brand, model, year, fuel type, and
kilometers driven. Click **Predict Price** for an instant ML estimate.
        """)

        # Dataset overview cards
        st.markdown("#### 📈 Dataset Coverage")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Listings", f"{len(car_df):,}")
        d2.metric("Countries",      car_df['country'].nunique() if 'country' in car_df.columns else 1)
        d3.metric("Brands",         car_df['company'].nunique())
        d4.metric("Year Range",     f"{int(car_df['year'].min())}–{int(car_df['year'].max())}")

        if is_global and 'country' in car_df.columns:
            st.markdown("#### 🌍 Listings by Country")
            by_country = (
                car_df['country'].value_counts()
                                 .reset_index()
                                 .rename(columns={'index': 'Country', 'country': 'Listings'})
                                 .head(15)
            )
            st.dataframe(by_country, use_container_width=True, hide_index=True)

with col_info:
    st.markdown("### 🤖 Model Info")
    if model_results:
        st.markdown(f"**Best:** `{model_results['best_name']}`")
        st.markdown(f"**R² Score:** `{model_results['best_score']:.4f}`")
        st.markdown(f"**Trained on:** `{model_results.get('n_samples', '?'):,}` samples")
        st.markdown("---")
        st.markdown("**All Models:**")
        for name, res in model_results['results'].items():
            with st.expander(name):
                st.write(f"R²:  `{res['r2']:.4f}`")
                st.write(f"MAE: `${res['mae']:,.0f}`")

    st.markdown("---")
    st.markdown("### 📌 Features Used")
    feats = ["Market / Country", "Model Year", "Brand", "Car Model",
             "Fuel Type", "Transmission", "Kms Driven"]
    for f in feats:
        st.markdown(f'<span class="feature-tag">{f}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline")
    st.markdown("""
1. **OHE** — brand, model, fuel, transmission, country
2. **StandardScaler** — year, kms
3. **80/20 split** — random_state=42
4. Best of 4 algorithms selected
    """)

    st.markdown("---")
    st.markdown("### 🌍 Data Sources")
    st.markdown("""
- Quikr India (original)
- Craigslist USA
- AutoScout24 Europe
- OLX multi-country
- Kaggle global datasets
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.8rem;'>"
    "Built with Streamlit · scikit-learn · Data from 15+ global sources"
    "</div>",
    unsafe_allow_html=True
)
