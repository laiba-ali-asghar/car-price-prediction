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
# Load model pipeline
# ===============================
@st.cache_resource
def load_pipeline():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

# ===============================
# Load dataset
# ===============================
df = pd.read_csv(DATA_PATH, na_values="?").dropna(subset=["price"])

# ===============================
# Identify features
# ===============================
NUMERICAL_FEATURES = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
NUMERICAL_FEATURES = [c for c in NUMERICAL_FEATURES if c != "price"]

CATEGORICAL_FEATURES = df.select_dtypes(include=["object"]).columns.tolist()

MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ===============================
# Streamlit UI
# ===============================
st.title("🚗 Car Price Prediction App")

st.markdown("Enter the car details below and get the predicted price.")

user_input = {}

st.subheader("Car Information")

cols = st.columns(2)

# Numeric Inputs
for idx, col in enumerate(NUMERICAL_FEATURES):
    with cols[idx % 2]:
        default_value = float(df[col].median())
        user_input[col] = st.number_input(
            label=col,
            value=default_value,
            format="%.2f"
        )

# Categorical Inputs
for idx, col in enumerate(CATEGORICAL_FEATURES):
    with cols[idx % 2]:
        options = sorted(df[col].dropna().unique().tolist())
        default = df[col].mode()[0]
        user_input[col] = st.selectbox(col, options, index=options.index(default))

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])
input_df = input_df[MODEL_FEATURES]  # ensure correct order

# ===============================
# Predict button
# ===============================
if st.button("Predict Price"):
    try:
        prediction = pipeline.predict(input_df)[0]

        st.markdown(
            f"""
            <div style="background-color:#1f4e79;padding:25px;border-radius:12px;text-align:center;">
                <h3 style="color:white;">Estimated Car Price</h3>
                <h1 style="color:#f9c74f;">${prediction:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("❌ Prediction failed. The model may have been trained on different features.")
        st.write("Error details:")
        st.write(str(e))
        st.write("Model expected features:")
        st.write(MODEL_FEATURES)
        st.write("Input sent to model:")
        st.write(input_df)
