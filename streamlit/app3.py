import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Google Trends Search Volume Predictor",
    page_icon="📈",
    layout="centered"
)

# -------------------------------
# Load Model & Encoder
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("a.pkl")
    encoder = joblib.load("b.pkl")
    return model, encoder

model, category_encoder = load_artifacts()

# -------------------------------
# App Header
# -------------------------------
st.title("📊 Google Trends Search Volume Predictor")
st.markdown(
    """
    Predict **search volume** using trend growth, category,
    and date-based features.

    **Model:** Random Forest Regressor  
    **Data Source:** Google Trends (US)
    """
)

st.divider()

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("🔢 Input Parameters")

date_input = st.date_input(
    "Select Date",
    value=datetime.today()
)

increase_pct = st.number_input(
    "Increase Percentage (%)",
    min_value=-100.0,
    max_value=500.0,
    value=10.0,
    step=0.1
)

# Categories pulled from encoder (SAFE & FUTURE-PROOF)
category = st.selectbox(
    "Select Search Category",
    sorted(category_encoder.classes_)
)

# -------------------------------
# Feature Engineering
# -------------------------------
year = date_input.year
month = date_input.month
day = date_input.day
dayofweek = date_input.weekday()

# Label Encoding (CORRECT way)
category_encoded = category_encoder.transform([category])[0]

# Final input dataframe (MUST match training)
input_df = pd.DataFrame({
    "increase_percentage": [increase_pct],
    "year": [year],
    "month": [month],
    "day": [day],
    "dayofweek": [dayofweek],
    "category_encoded": [category_encoded]
})

# -------------------------------
# Prediction
# -------------------------------
st.divider()

if st.button("🚀 Predict Search Volume"):
    prediction = model.predict(input_df)[0]

    st.success("Prediction Successful ✅")
    st.metric(
        label="📈 Predicted Search Volume",
        value=f"{int(prediction):,}"
    )

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("Random Forest Model | Google Trends US Data")

