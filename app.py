import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="House Price AI", page_icon="🏠", layout="centered")

# 2. Load the Model
@st.cache_resource
def load_model():
    # Looks in the same folder as app.py
    model_path = os.path.join(os.path.dirname(__file__), "house_price_pipeline.pkl")
    return joblib.load(model_path)

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. UI Header
st.title("🏠 AI Property Valuation")
st.markdown("---")

# 4. User Inputs
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
    gr_liv_area = st.number_input("Living Area (sq.ft)", 100, 10000, 1500)
    garage_cars = st.number_input("Garage Capacity (cars)", 0, 5, 2)

with col2:
    total_bsmt_sf = st.number_input("Basement Area (sq.ft)", 0, 5000, 1000)
    full_bath = st.number_input("Full Bathrooms", 1, 5, 2)
    year_built = st.number_input("Year Built", 1850, 2026, 1995)

# 5. Prediction Logic
if st.button("Calculate Estimate", type="primary", use_container_width=True):
    # Match the exact names your model expects
    input_df = pd.DataFrame([{
        "Overall Qual": overall_qual,
        "Gr Liv Area": gr_liv_area,
        "Garage Cars": garage_cars,
        "Total Bsmt SF": total_bsmt_sf,
        "Full Bath": full_bath,
        "Year Built": year_built
    }])
    
    with st.spinner("Analyzing market data..."):
        pred_log = pipeline.predict(input_df)
        # Reverse log transformation
        final_price = np.expm1(pred_log)[0]
    
    st.balloons()
    st.metric(label="Estimated Market Value", value=f"${final_price:,.2f}")
    st.caption(f"Approximate value in INR: ₹{final_price * 90:,.2f}")