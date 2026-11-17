import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="House Price AI",
    page_icon="🏠",
    layout="centered"
)

# 2. Load the Model
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "house_price_pipeline.pkl"
    )
    return joblib.load(model_path)


try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Unable to load the ML model: {e}")
    st.stop()


# 3. UI Header
st.title("🏠 AI Property Valuation")
st.write(
    "Estimate property prices using a Machine Learning model."
)
st.markdown("---")


# 4. User Inputs
current_year = datetime.now().year

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider(
        "Overall Quality (1-10)", 1, 10, 6
    )

    gr_liv_area = st.number_input(
        "Living Area (sq.ft)",
        min_value=100,
        max_value=10000,
        value=1500
    )

    garage_cars = st.number_input(
        "Garage Capacity (cars)",
        min_value=0,
        max_value=5,
        value=2
    )

with col2:
    total_bsmt_sf = st.number_input(
        "Basement Area (sq.ft)",
        min_value=0,
        max_value=5000,
        value=1000
    )

    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1850,
        max_value=current_year,
        value=1995
    )


# 5. Prediction Logic
if st.button(
    "Calculate Estimate",
    type="primary",
    use_container_width=True
):

    # Prepare input with exact training feature names
    input_df = pd.DataFrame([{
        "Overall Qual": overall_qual,
        "Gr Liv Area": gr_liv_area,
        "Garage Cars": garage_cars,
        "Total Bsmt SF": total_bsmt_sf,
        "Full Bath": full_bath,
        "Year Built": year_built
    }])

    try:
        with st.spinner("Analyzing property data..."):

            # Model prediction (log-transformed)
            pred_log = pipeline.predict(input_df)

            # Reverse log transformation
            final_price = float(np.expm1(pred_log[0]))

        if not np.isfinite(final_price) or final_price <= 0:
            st.error("Invalid prediction. Please check your inputs.")
            st.stop()

        # 6. Display Results
        st.success("Property valuation completed!")

        st.metric(
            label="Estimated Property Value",
            value=f"${final_price:,.2f}"
        )

        st.caption(
            "Prediction generated using the trained ML pipeline."
        )

        # Display entered property details
        with st.expander("View Property Details"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
