import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# Paths
# ==============================
MODEL_PATH = "car_price_model.pkl"
DATA_PATH = "carprice.csv"

# ==============================
# Load model
# ==============================
@st.cache_resource
def load_pipeline():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

# ==============================
# Load training dataset
# ==============================
df = pd.read_csv(DATA_PATH, na_values="?").dropna(subset=["price"])

# BELOW IS CRITICAL:
# Convert all categorical columns to category codes (same encoding as training)
cat_cols = df.select_dtypes(include="object").columns.tolist()

for col in cat_cols:
    df[col] = df[col].astype("category")
    df[col] = df[col].cat.codes

# This gives the model's exact expected structure
MODEL_FEATURES = pipeline.feature_names_in_.tolist()

# ==============================
# Streamlit UI
# ==============================
st.title("🚗 Car Price Prediction App")

st.write("Fill in the car details to predict the selling price.")

user_data = {}

cols = st.columns(2)

for idx, col in enumerate(MODEL_FEATURES):

    if col in cat_cols:
        # categorical → show dropdown of original labels
        original_categories = pd.read_csv(DATA_PATH, na_values="?")[col].dropna().unique().tolist()
        default = pd.read_csv(DATA_PATH, na_values="?")[col].mode()[0]

        with cols[idx % 2]:
            choice = st.selectbox(col, original_categories, index=original_categories.index(default))

            # encode using SAME mapping
            df_original = pd.read_csv(DATA_PATH, na_values="?")
            df_original[col] = df_original[col].astype("category")
            code_map = dict(enumerate(df_original[col].cat.categories))

            # reverse mapping (label → code)
            reverse_map = {v: k for k, v in code_map.items()}

            user_data[col] = reverse_map[choice]

    else:
        # numeric feature
        default = float(df[col].median())
        with cols[idx % 2]:
            user_data[col] = st.number_input(col, value=default)

# Build final input row
input_df = pd.DataFrame([user_data])
input_df = input_df[MODEL_FEATURES]

# ==============================
# Predict
# ==============================
if st.button("Predict Price"):
    try:
        pred = pipeline.predict(input_df)[0]
        st.success(f"Predicted Car Price: ${pred:,.2f}")

    except Exception as e:
        st.error("Prediction failed.")
        st.write(str(e))
        st.write("Input sent to model:")
        st.write(input_df)
