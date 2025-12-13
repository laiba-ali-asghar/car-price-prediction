import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================================
# Streamlit Page Configuration
# ==========================================================
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# ==========================================================
# File Paths (MUST match your saved files)
# ==========================================================
MODEL_PATH = "car_price_model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"
DATA_PATH = "carprice.csv"

# ==========================================================
# Load Model & Preprocessor (cached)
# ==========================================================
@st.cache_resource
def load_assets():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except FileNotFoundError:
        st.error("Required model files not found. Make sure model and preprocessor exist.")
        return None, None

model, preprocessor = load_assets()
if model is None or preprocessor is None:
    st.stop()

# ==========================================================
# Load Raw Dataset (only for UI defaults & categories)
# ==========================================================
try:
    df_raw = pd.read_csv(DATA_PATH, na_values='?')
    df_raw = df_raw.dropna(subset=['price'])
except FileNotFoundError:
    st.error("Dataset file 'carprice.csv' not found.")
    st.stop()

# ==========================================================
# Extract feature lists DIRECTLY from preprocessor (FIX)
# ==========================================================
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]

# ==========================================================
# Mappings used in training
# ==========================================================
WORD_TO_NUM = {
    'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'eight': 8, 'twelve': 12
}

# ==========================================================
# Prediction Function (ROBUST)
# ==========================================================
def predict_price(user_input: dict):
    # Create template row from training data
    template = df_raw.drop(['price', 'symboling'], axis=1).iloc[0].copy()

    # Replace template values with user input
    for key, value in user_input.items():
        template[key] = value

    input_df = pd.DataFrame([template])

    # Convert textual numeric columns
    if 'num-of-cylinders' in input_df.columns:
        input_df['num-of-cylinders'] = input_df['num-of-cylinders'].map(WORD_TO_NUM)
    if 'num-of-doors' in input_df.columns:
        input_df['num-of-doors'] = input_df['num-of-doors'].map({'two': 2, 'four': 4})

    # Apply preprocessing and predict
    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)
    return prediction[0]

# ==========================================================
# UI – TITLE & INTRO
# ==========================================================
st.title("🚗 Car Price Prediction System")
st.markdown("""
This application predicts the **selling price of a car** using a machine learning regression model.
The model was trained using **EDA, preprocessing pipelines, and Random Forest regression**.
""")
st.divider()

# ==========================================================
# UI – INPUT SECTION
# ==========================================================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Enter Car Specifications")

    user_input = {}

    # Categorical Inputs
    for col in ['make', 'body-style', 'drive-wheels', 'fuel-type']:
        user_input[col] = st.selectbox(
            col.replace('-', ' ').title(),
            sorted(df_raw[col].dropna().unique().tolist())
        )

    # Numerical Inputs (important predictors)
    user_input['horsepower'] = st.slider("Horsepower", 40, 300, 100)
    user_input['engine-size'] = st.slider("Engine Size (cc)", 60, 350, 120)
    user_input['curb-weight'] = st.slider("Curb Weight (lbs)", 1400, 4100, 2500)
    user_input['highway-mpg'] = st.slider("Highway MPG", 15, 60, 30)

with col2:
    st.subheader("Prediction Output")
    st.markdown("Click the button below to predict the car price.")

    if st.button("Predict Price", type="primary"):
        price = predict_price(user_input)
        st.markdown(
            f"""
            <div style="background-color:#1f4e79;padding:25px;border-radius:12px;text-align:center;">
                <h3 style="color:white;">Estimated Car Price</h3>
                <h1 style="color:#f9c74f;">${price:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Adjust inputs and press **Predict Price**.")

# ==========================================================
# UI – MODEL INFO
# ==========================================================
st.divider()
st.header("📌 Model Information")
st.markdown("""
- **Algorithm:** Random Forest Regressor  
- **Evaluation Metric:** RMSE & R² Score  
- **R² ≈ 0.90** → Explains ~90% variance in car prices  
- **RMSE ≈ $2,400** average prediction error

**Why this works well:**
- Handles non‑linear relationships
- Robust to outliers
- No feature scaling issues
""")

# ==========================================================
# END
# ==========================================================
