import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ======================
# Paths
# ======================
MODEL_PATH = "car_price_model.pkl"
DATA_PATH = "carprice.csv"

# ======================
# Load model
# ======================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ======================
# Load dataset
# ======================
df = pd.read_csv(DATA_PATH, na_values="?").dropna()

# ======================
# Columns (from your model)
# ======================
MODEL_FEATURES = [
    "symboling","normalized-losses","wheel-base","length","width","height",
    "curb-weight","engine-size","bore","stroke","compression-ratio","horsepower",
    "peak-rpm","city-mpg","highway-mpg","make","fuel-type","aspiration",
    "num-of-doors","body-style","drive-wheels","engine-location","engine-type",
    "num-of-cylinders","fuel-system"
]

NUMERICAL = [
    "symboling","normalized-losses","wheel-base","length","width","height",
    "curb-weight","engine-size","bore","stroke","compression-ratio","horsepower",
    "peak-rpm","city-mpg","highway-mpg"
]

CATEGORICAL = [
    "make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels",
    "engine-location","engine-type","num-of-cylinders","fuel-system"
]

# ======================
# Build label encoders from training CSV
# ======================
label_maps = {}

for col in CATEGORICAL:
    df[col] = df[col].astype("category")
    # mapping: category_name -> code
    label_maps[col] = {cat: code for code, cat in enumerate(df[col].cat.categories)}

# ======================
# UI
# ======================
st.title("🚗 Car Price Prediction")

inputs = {}
cols = st.columns(2)

# numeric inputs
for i, col in enumerate(NUMERICAL):
    default = float(df[col].median())
    with cols[i % 2]:
        inputs[col] = st.number_input(col, value=default)

# categorical inputs
for i, col in enumerate(CATEGORICAL):
    with cols[i % 2]:
        options = list(label_maps[col].keys())
        default = options[0]
        choice = st.selectbox(col, options)
        inputs[col] = label_maps[col][choice]   # convert to integer code

# build dataframe
input_df = pd.DataFrame([inputs])[MODEL_FEATURES]

# ======================
# Prediction
# ======================
if st.button("Predict Price"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Price: ${pred:,.2f}")

    except Exception as e:
        st.error("Prediction failed.")
        st.write(str(e))
        st.write("Input sent to model:")
        st.write(input_df)
