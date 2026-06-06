import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    .price-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .price-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .price-usd {
        font-size: 1.2rem;
        opacity: 0.85;
    }
    .metric-card {
        background: #f8f9ff;
        border: 1px solid #e8eaf6;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .feature-tag {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    .sidebar .sidebar-content {
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model and data ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_car.csv')

@st.cache_data
def load_results():
    try:
        with open('model_results.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()
car_df = load_data()
model_results = load_results()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🚗 Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Estimate the resale value of any used car using Machine Learning</div>', unsafe_allow_html=True)

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Vehicle Details")
    st.markdown("---")

    companies = sorted(car_df['company'].unique())
    selected_company = st.selectbox("🏢 Brand / Company", companies, index=companies.index('Hyundai') if 'Hyundai' in companies else 0)

    company_models = sorted(car_df[car_df['company'] == selected_company]['name'].unique())
    selected_model = st.selectbox("🚘 Car Model", company_models)

    years = sorted(car_df['year'].unique(), reverse=True)
    selected_year = st.selectbox("📅 Model Year", years)

    fuel_types = sorted(car_df['fuel_type'].unique())
    selected_fuel = st.selectbox("⛽ Fuel Type", fuel_types)

    kms_driven = st.number_input(
        "🛣️ Kilometers Driven",
        min_value=0,
        max_value=1000000,
        value=30000,
        step=1000,
        help="Total distance the car has been driven"
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)

# ── Main content ───────────────────────────────────────────────────────────────
col_main, col_info = st.columns([2, 1])

with col_main:
    if predict_btn:
        input_data = pd.DataFrame({
            'name': [selected_model],
            'company': [selected_company],
            'year': [selected_year],
            'kms_driven': [kms_driven],
            'fuel_type': [selected_fuel]
        })

        predicted_price = model.predict(input_data)[0]
        predicted_price = max(0, predicted_price)

        st.markdown("### 💰 Prediction Result")
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated Resale Value</div>
            <div class="price-value">₹ {predicted_price:,.0f}</div>
            <div class="price-usd">≈ ${predicted_price / 83:,.0f} USD</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📋 Your Vehicle Summary")
        c1, c2, c3 = st.columns(3)
        car_age = 2024 - selected_year
        c1.metric("Car Age", f"{car_age} years")
        c2.metric("Kms Driven", f"{kms_driven:,} km")
        c3.metric("Fuel Type", selected_fuel)

        # Similar cars price range
        similar = car_df[
            (car_df['company'] == selected_company) &
            (car_df['year'].between(selected_year - 2, selected_year + 2))
        ]['Price']

        if not similar.empty:
            st.markdown("#### 📊 Market Price Range (Similar Cars)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Min Price", f"₹ {similar.min():,.0f}")
            m2.metric("Avg Price", f"₹ {similar.mean():,.0f}")
            m3.metric("Max Price", f"₹ {similar.max():,.0f}")

        # Depreciation insight
        age = 2024 - selected_year
        if age < 3:
            dep_msg = "🟢 Low depreciation — relatively new vehicle"
        elif age < 7:
            dep_msg = "🟡 Moderate depreciation — mid-aged vehicle"
        else:
            dep_msg = "🔴 High depreciation — older vehicle"

        km_msg = "🟢 Low mileage" if kms_driven < 40000 else ("🟡 Moderate mileage" if kms_driven < 80000 else "🔴 High mileage")

        st.info(f"{dep_msg}  |  {km_msg}")

    else:
        st.markdown("### 👈 Enter vehicle details in the sidebar")
        st.markdown("""
        **How it works:**
        1. Select your car's **brand**, **model**, and **year**
        2. Choose the **fuel type**
        3. Enter total **kilometers driven**
        4. Click **Predict Price** for an instant estimate

        The model uses **machine learning** trained on thousands of real used car transactions.
        """)

        # Dataset overview
        st.markdown("#### 📈 Dataset Overview")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Cars", f"{len(car_df):,}")
        d2.metric("Brands", car_df['company'].nunique())
        d3.metric("Models", car_df['name'].nunique())
        d4.metric("Fuel Types", car_df['fuel_type'].nunique())

with col_info:
    st.markdown("### 🤖 Model Info")

    if model_results:
        st.markdown(f"**Best Model:** {model_results['best_name']}")
        st.markdown(f"**R² Score:** `{model_results['best_score']:.4f}`")
        st.markdown("---")
        st.markdown("**All Models Performance:**")
        for name, res in model_results['results'].items():
            with st.expander(name):
                st.write(f"R² Score: `{res['r2']:.4f}`")
                st.write(f"MAE: ₹ `{res['mae']:,.0f}`")

    st.markdown("---")
    st.markdown("### 📌 Key Features Used")
    features = ["Model Year", "Brand / Company", "Car Model", "Fuel Type", "Kms Driven"]
    for f in features:
        st.markdown(f'<span class="feature-tag">{f}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline")
    st.markdown("""
    1. **One-Hot Encoding** for brand, model, fuel type
    2. **Standard Scaling** for year & kms
    3. **80/20 Train-Test Split**
    4. Best of 3 algorithms selected
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Built with Streamlit · scikit-learn · Powered by ML"
    "</div>",
    unsafe_allow_html=True
)
