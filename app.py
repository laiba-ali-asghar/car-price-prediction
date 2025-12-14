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
# Feature lists
# ===============================
NUMERICAL_FEATURES = [
    'symboling','normalized-losses','wheel-base','length','width','height','curb-weight',
    'engine-size','bore','stroke','compression-ratio','horsepower','peak-rpm',
    'city-mpg','highway-mpg','num-of-doors','num-of-cylinders'
]

CATEGORICAL_FEATURES = [
    'make','fuel-type','aspiration','body-style','drive-wheels',
    'engine-location','engine-type','fuel-system'
]

MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ===============================
# Load pipeline
# ===============================
@st.cache_resource
def load_pipeline():
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_pipeline()

# ===============================
# Load data for defaults
# ===============================
df = pd.read_csv(DATA_PATH, na_values="?").dropna(subset=['price'])

# ===============================
# Default input
# ===============================
defaults = {
    'make': df['make'].mode()[0],
    'fuel-type': df['fuel-type'].mode()[0],
    'aspiration': df['aspiration'].mode()[0],
    'body-style': df['body-style'].mode()[0],
    'drive-wheels': df['drive-wheels'].mode()[0],
    'horsepower': float(df['horsepower'].median()),
    'engine-size': float(df['engine-size'].median()),
    'curb-weight': float(df['curb-weight'].median()),
    'highway-mpg': float(df['highway-mpg'].median()),
    'symboling': 0,
    'normalized-losses': float(df['normalized-losses'].median()) if 'normalized-losses' in df else 0,
    'engine-location': 'front',
    'wheel-base': float(df['wheel-base'].median()),
    'length': float(df['length'].median()),
    'width': float(df['width'].median()),
    'height': float(df['height'].median()),
    'engine-type': df['engine-type'].mode()[0],
    'fuel-system': df['fuel-system'].mode()[0],
    'bore': float(df['bore'].median()),
    'stroke': float(df['stroke'].median()),
    'compression-ratio': float(df['compression-ratio'].median()),
    'peak-rpm': float(df['peak-rpm'].median()),
    'city-mpg': float(df['city-mpg'].median()),
    'num-of-doors': 'four',
    'num-of-cylinders': 'four'
}

# Map word numbers to digits
word_to_digit = {'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}
defaults['num-of-doors'] = word_to_digit[defaults['num-of-doors']]
defaults['num-of-cylinders'] = word_to_digit[defaults['num-of-cylinders']]

# Convert to DataFrame
input_df = pd.DataFrame([defaults])

# Ensure correct dtypes
input_df[NUMERICAL_FEATURES] = input_df[NUMERICAL_FEATURES].astype(float)
input_df[CATEGORICAL_FEATURES] = input_df[CATEGORICAL_FEATURES].astype(object)

# ===============================
# Streamlit UI
# ===============================
st.title("🚗 Car Price Prediction")
st.markdown("### Predicted Price using Default Values")

# Predict
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
