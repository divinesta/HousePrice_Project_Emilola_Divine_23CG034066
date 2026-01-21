"""
House Price Prediction - Streamlit Version
Alternative web interface using Streamlit

To run: streamlit run app_streamlit.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
    }
    .prediction-price {
        font-size: 3em;
        font-weight: bold;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects


@st.cache_resource
def load_model_objects():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('model/house_price_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        label_encoder = joblib.load('model/label_encoder.pkl')
        return model, scaler, label_encoder, None
    except Exception as e:
        return None, None, None, str(e)


model, scaler, label_encoder, error = load_model_objects()

# Header
st.title("🏠 House Price Prediction System")
st.markdown("### Predict house prices using machine learning")
st.markdown("---")

if error:
    st.error(f"❌ Error loading model: {error}")
    st.info("Please ensure the model files (.pkl) are in the 'model' folder.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📊 About the Model")
    st.info("""
    **Algorithm**: Random Forest Regressor
    
    **Features Used**:
    - Overall Quality
    - Living Area
    - Basement Area
    - Garage Cars
    - Year Built
    - Neighborhood
    
    **Model Persistence**: Joblib
    """)

    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Enter house details in the form
    2. Click 'Predict Price'
    3. View the estimated price
    """)

# Main form
neighborhoods = sorted(label_encoder.classes_.tolist())

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏗️ House Characteristics")

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=7,
        help="Quality of materials and finish (1=Poor, 10=Excellent)"
    )

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=0.0,
        max_value=10000.0,
        value=1500.0,
        step=50.0,
        help="Above grade living area in square feet"
    )

    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=50.0,
        help="Total basement area in square feet"
    )

with col2:
    st.subheader("🚗 Additional Features")

    garage_cars = st.selectbox(
        "Garage Size (cars)",
        options=[0, 1, 2, 3, 4, 5],
        index=2,
        help="Garage capacity in number of cars"
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000,
        step=1,
        help="Original construction year"
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        options=neighborhoods,
        help="Physical location within city limits"
    )

# Prediction button
st.markdown("---")
if st.button("🔮 Predict House Price", type="primary"):
    try:
        # Encode neighborhood
        neighborhood_encoded = label_encoder.transform([neighborhood])[0]

        # Prepare features
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            neighborhood_encoded
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Display result
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: white;">Predicted House Price</h2>
            <div class="prediction-price">${prediction:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Display input summary
        st.markdown("---")
        st.subheader("📋 Input Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Quality", overall_qual)
            st.metric("Living Area", f"{gr_liv_area:,.0f} sq ft")
        with col2:
            st.metric("Basement Area", f"{total_bsmt_sf:,.0f} sq ft")
            st.metric("Garage Cars", garage_cars)
        with col3:
            st.metric("Year Built", year_built)
            st.metric("Neighborhood", neighborhood)

        # Success message
        st.success("✅ Prediction completed successfully!")

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Machine Learning Model:</strong> Random Forest Regressor</p>
    <p>© 2026 House Price Prediction System</p>
</div>
""", unsafe_allow_html=True)
