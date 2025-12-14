import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
# Feature Lists
# ======================
NUMERICAL = [
    "symboling","normalized-losses","wheel-base","length","width","height",
    "curb-weight","engine-size","bore","stroke","compression-ratio","horsepower",
    "peak-rpm","city-mpg","highway-mpg"
]

CATEGORICAL = [
    "make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels",
    "engine-location","engine-type","num-of-cylinders","fuel-system"
]

ALL_FEATURES = NUMERICAL + CATEGORICAL

# ======================
# Build Preprocessor (REQUIRED!)
# ======================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERICAL)
    ]
)

# ======================
# Streamlit UI
# ======================
st.title("🚗 Car Price Prediction App")

inputs = {}

st.subheader("Enter Car Specifications")

cols = st.columns(2)

# numeric fields
for i, col in enumerate(NUMERICAL):
    with cols[i % 2]:
        default = float(df[col].median())
        inputs[col] = st.number_input(col, value=default)

# categorical fields
for i, col in enumerate(CATEGORICAL):
    with cols[i % 2]:
        choices = sorted(df[col].unique().tolist())
        default = df[col].mode()[0]
        inputs[col] = st.selectbox(col, choices, index=choices.index(default))

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# ======================
# Predict
# ======================
if st.button("Predict Price"):

    try:
        # apply preprocessing (OneHotEncoder)
        X_processed = preprocessor.fit(df[ALL_FEATURES]).transform(input_df)

        # run final model prediction
        pred = model.predict(X_processed)[0]

        st.success(f"Predicted Price: ${pred:,.2f}")

    except Exception as e:
        st.error("Prediction failed.")
        st.write(str(e))
        st.write("Input:", input_df)
