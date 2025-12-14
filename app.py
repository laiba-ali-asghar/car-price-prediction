import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# ==========================================================
# File paths
# ==========================================================
MODEL_PATH = "car_price_model.pkl"
DATA_PATH = "carprice.csv"

# ==========================================================
# Features (must match training pipeline)
# ==========================================================
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

# ==========================================================
# Load pipeline and data
# ==========================================================
@st.cache_resource
def load_pipeline_and_data():
    try:
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        df_raw = pd.read_csv(DATA_PATH, na_values="?")
        df = df_raw.dropna(subset=['price']).copy()
        # Ensure numeric-like columns are numeric
        for col in ['bore', 'stroke', 'horsepower', 'peak-rpm']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return pipeline, df
    except Exception as e:
        st.error(f"Error loading pipeline or data: {e}")
        st.stop()

pipeline, df_cleaned = load_pipeline_and_data()

# ==========================================================
# Default values
# ==========================================================
defaults = {
    'make': df_cleaned['make'].mode()[0],
    'fuel-type': df_cleaned['fuel-type'].mode()[0],
    'aspiration': df_cleaned['aspiration'].mode()[0],
    'body-style': df_cleaned['body-style'].mode()[0],
    'drive-wheels': df_cleaned['drive-wheels'].mode()[0],
    'horsepower': int(df_cleaned['horsepower'].median()),
    'engine-size': int(df_cleaned['engine-size'].median()),
    'curb-weight': int(df_cleaned['curb-weight'].median()),
    'highway-mpg': int(df_cleaned['highway-mpg'].median())
}

hidden_features = {
    'symboling': 0,
    'normalized-losses': np.nan,
    'engine-location': 'front',
    'wheel-base': df_cleaned['wheel-base'].median(),
    'length': df_cleaned['length'].median(),
    'width': df_cleaned['width'].median(),
    'height': df_cleaned['height'].median(),
    'engine-type': df_cleaned['engine-type'].mode()[0],
    'fuel-system': df_cleaned['fuel-system'].mode()[0],
    'bore': df_cleaned['bore'].median(),
    'stroke': df_cleaned['stroke'].median(),
    'compression-ratio': df_cleaned['compression-ratio'].median(),
    'peak-rpm': df_cleaned['peak-rpm'].median(),
    'city-mpg': df_cleaned['city-mpg'].median(),
    'num-of-doors': 'four',
    'num-of-cylinders': 'four'
}

# ==========================================================
# Combine defaults
# ==========================================================
all_input = {**defaults, **hidden_features}

# Convert to DataFrame
input_df = pd.DataFrame([all_input])

# Map word numbers to digits
word_to_digit = {'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}
for col in ['num-of-doors','num-of-cylinders']:
    input_df[col] = input_df[col].astype(str).str.lower().map(word_to_digit)

# Ensure numeric and categorical types
input_df[NUMERICAL_FEATURES] = input_df[NUMERICAL_FEATURES].astype(float)
input_df[CATEGORICAL_FEATURES] = input_df[CATEGORICAL_FEATURES].astype(object)

# ==========================================================
# Show default prediction
# ==========================================================
st.title("🚗 Car Price Prediction")
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

st.divider()

# ==========================================================
# Adjust important features
# ==========================================================
st.header("🔧 Adjust Car Features (Optional)")
col1, col2, col3 = st.columns(3)

user_input = {}
user_input['make'] = col1.selectbox("Make", df_cleaned['make'].unique(), index=0)
user_input['body-style'] = col2.selectbox("Body Style", df_cleaned['body-style'].unique(), index=0)
user_input['drive-wheels'] = col3.selectbox("Drive Wheels", df_cleaned['drive-wheels'].unique(), index=0)

col4, col5, col6 = st.columns(3)
user_input['horsepower'] = col4.slider("Horsepower (hp)", 48, 288, defaults['horsepower'])
user_input['engine-size'] = col5.slider("Engine Size (cc)", 61, 326, defaults['engine-size'])
user_input['curb-weight'] = col6.slider("Curb Weight (lbs)", 1488, 4066, defaults['curb-weight'])

user_input.update(hidden_features)  # add hidden features

# Predict button
if st.button("Predict with Adjusted Features"):
    user_df = pd.DataFrame([user_input])
    for col in ['num-of-doors','num-of-cylinders']:
        user_df[col] = user_df[col].astype(str).str.lower().map(word_to_digit)
    user_df[NUMERICAL_FEATURES] = user_df[NUMERICAL_FEATURES].astype(float)
    user_df[CATEGORICAL_FEATURES] = user_df[CATEGORICAL_FEATURES].astype(object)
    predicted_price_adj = pipeline.predict(user_df)[0]

    st.markdown(
        f"""
        <div style="background-color:#1f4e79;padding:25px;border-radius:12px;text-align:center;">
            <h3 style="color:white;">Adjusted Predicted Price</h3>
            <h1 style="color:#f9c74f;">${predicted_price_adj:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ==========================================================
# Project Stats / Info
# ==========================================================
st.header("📊 Project & Model Stats")

col_model, col_conclusion = st.columns(2)

with col_model:
    st.subheader("Model Performance")
    RMSE_VALUE = 2504.66
    R2_VALUE = 0.9009
    st.metric("Root Mean Squared Error (RMSE)", f"${RMSE_VALUE:,.2f}")
    st.metric("R-squared (R²)", f"{R2_VALUE:.4f}")
    st.markdown("""
    - Model: Random Forest Regressor
    - Pipeline: One-Hot Encoding + Scaling
    """)

with col_conclusion:
    st.subheader("Conclusion")
    st.markdown("""
    - Random Forest achieved high performance (R² ≈ 0.90).  
    - Application demonstrates full workflow: EDA → Preprocessing → Prediction.  
    - Default prediction is shown first for convenience.
    """)

plt.close('all')
