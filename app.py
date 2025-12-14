import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# File paths
# ===============================
MODEL_PATH = "car_price_model.pkl"
DATA_PATH = "carprice.csv"

# ===============================
# Load pipeline
# ===============================
@st.cache_resource
def load_pipeline():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

# The model expects these exact columns:
MODEL_FEATURES = pipeline.feature_names_in_.tolist()

# Load dataset for default values
df = pd.read_csv(DATA_PATH, na_values="?").dropna(subset=['price'])

# ===============================
# Build default input values
# ===============================
defaults = {}

for col in MODEL_FEATURES:

    # Numerical feature
    if pd.api.types.is_numeric_dtype(df[col]):
        defaults[col] = float(df[col].median())

    # Categorical feature
    else:
        defaults[col] = df[col].mode()[0]

# Convert to DataFrame
input_df = pd.DataFrame([defaults])

# Final reorder to match pipeline
input_df = input_df[pipeline.feature_names_in_]

# ===============================
# Streamlit UI
# ===============================
st.title("🚗 Car Price Prediction")
st.markdown("### Predicted Price using Default Values")

try:
    predicted_price = pipeline.predict(input_df)[0]

    st.markdown(
        f"""
        <div style="background-color:#1f4e79;padding:25px;border-radius:12px;text-align:center;">
            <h3 style="color:white;">Estimated Selling Price</h3>
            <h1 style="color:#f9c74f;">${predicted_price:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

except Exception as e:
    st.error("❌ Prediction failed. Your model and input feature columns do not match.")
    st.write("### Expected columns by the model:")
    st.write(list(pipeline.feature_names_in_))
    st.write("### Columns provided to model:")
    st.write(list(input_df.columns))
    st.write("### Error message:")
    st.write(str(e))
