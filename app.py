import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# Streamlit Page Configuration
# ==========================================================
st.set_page_config(page_title="Car Price Prediction Project", layout="wide")

# ==========================================================
# File Paths
# ==========================================================
MODEL_PATH = "car_price_model.pkl"
DATA_PATH = "carprice.csv"

# ==========================================================
# Model Feature Lists (CRITICAL FIX: This order MUST match the ColumnTransformer)
# Based on the metadata from your preprocessor.pkl, these are the expected lists:
# ==========================================================
# Features that go through the NUMERICAL transformer (17 features)
NUMERICAL_FEATURES = [
    # symboling is an integer column, likely added at the start of the numerical data
    'symboling',
    'normalized-losses', 
    'wheel-base', 
    'length', 
    'width', 
    'height', 
    'curb-weight', 
    'engine-size', 
    'bore', 
    'stroke', 
    'compression-ratio', 
    'horsepower', 
    'peak-rpm', 
    'city-mpg', 
    'highway-mpg',
    # num-of-doors and num-of-cylinders are converted to numbers (floats)
    'num-of-doors', 
    'num-of-cylinders'
]

# Features that go through the CATEGORICAL transformer (8 features)
CATEGORICAL_FEATURES = [
    'make', 'fuel-type', 'aspiration', 'body-style', 'drive-wheels', 
    'engine-location', 'engine-type', 'fuel-system'
]

# Combine all features: NUMERICAL_FEATURES first, then CATEGORICAL_FEATURES
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES 


# ==========================================================
# Load Pipeline & Data (Cached)
# ==========================================================
@st.cache_resource
def load_pipeline_and_data():
    """Load the full scikit-learn pipeline (model + preprocessor) and raw data."""
    try:
        # The car_price_model.pkl file contains the full Pipeline object
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        
        # Load data for defaults/EDA (treating '?' as NaN)
        df_raw = pd.read_csv(DATA_PATH, na_values="?")
        
        # Simple cleaning for EDA (drop price NaNs, convert known numerical-like columns for UI)
        df = df_raw.dropna(subset=['price']).copy()
        
        num_like_cols = ['bore', 'stroke', 'horsepower', 'peak-rpm']
        for col in num_like_cols:
            if col in df.columns:
                 # Ensure these UI default columns are numeric
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 
        return pipeline, df
    except FileNotFoundError:
        st.error(f"Required file not found: {MODEL_PATH} or {DATA_PATH}. Please ensure these are in your repository.")
        st.stop()
    except Exception as e:
        # The version error check remains important here
        st.error(f"Error loading files. Check scikit-learn version in requirements.txt (e.g., scikit-learn==1.7.2): {e}")
        st.stop()

pipeline, df_cleaned = load_pipeline_and_data()

# ==========================================================
# UI – INTRODUCTION & EDA
# ==========================================================
st.title("🚗 Data Science Capstone: Car Price Prediction")
st.markdown("""
This project predicts a car's selling price using a **Random Forest Regressor** pipeline.
""")

st.divider()

# --- EDA Section ---
st.header("📊 Exploratory Data Analysis & Insights")

st.subheader("1. Feature Correlation with Price")

fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
corr_matrix = df_cleaned[numeric_cols].corr()
sns.heatmap(corr_matrix[['price']].sort_values(by='price', ascending=False).head(8), 
            annot=True, fmt=".2f", cmap='coolwarm', cbar=False, ax=ax)
ax.set_title('Top Feature Correlation with Price')
st.pyplot(fig)

st.markdown("""
**Key Insight:** Features related to engine size and power (**Engine Size**, **Curb Weight**, **Horsepower**) show the highest positive correlation with price.
""")

st.divider()

# ==========================================================
# UI – PREDICTION INTERFACE
# ==========================================================
st.header("🔮 Interactive Price Prediction")
st.markdown("Select features below to get a real-time price prediction.")

# --- UI Defaults ---
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

user_input = {}
col_input, col_output = st.columns([2, 1])

with col_input:
    st.markdown("##### Adjust Car Specifications")
    
    # Row 1: Key Categorical Features
    col1, col2, col3 = st.columns(3)
    user_input['make'] = col1.selectbox("Make", df_cleaned['make'].unique(), index=list(df_cleaned['make'].unique()).index(defaults['make']))
    user_input['body-style'] = col2.selectbox("Body Style", df_cleaned['body-style'].unique(), index=list(df_cleaned['body-style'].unique()).index(defaults['body-style']))
    user_input['drive-wheels'] = col3.selectbox("Drive Wheels", df_cleaned['drive-wheels'].unique(), index=list(df_cleaned['drive-wheels'].unique()).index(defaults['drive-wheels']))

    # Row 2: Key Numerical Features (Sliders)
    user_input['horsepower'] = st.slider("Horsepower (hp)", 48, 288, defaults['horsepower'])
    user_input['engine-size'] = st.slider("Engine Size (cc)", 61, 326, defaults['engine-size'])
    user_input['curb-weight'] = st.slider("Curb Weight (lbs)", 1488, 4066, defaults['curb-weight'])
    user_input['highway-mpg'] = st.slider("Highway MPG", 16, 54, defaults['highway-mpg'])
    
    # Row 3: Other Categorical Features
    col4, col5, col6, col7 = st.columns(4)
    user_input['fuel-type'] = col4.radio("Fuel Type", df_cleaned['fuel-type'].unique(), index=list(df_cleaned['fuel-type'].unique()).index(defaults['fuel-type']))
    user_input['aspiration'] = col5.radio("Aspiration", df_cleaned['aspiration'].unique(), index=list(df_cleaned['aspiration'].unique()).index(defaults['aspiration']))
    # These must remain as strings 'four'/'two' here to be mapped to numbers later
    user_input['num-of-doors'] = col6.selectbox("Doors (Word)", ['four', 'two'], index=0)
    user_input['num-of-cylinders'] = col7.selectbox("Cylinders (Word)", ['four', 'six', 'five', 'eight', 'two', 'three', 'twelve'], index=0)

    # --- Hidden Imputed Values (Needed to complete the 25 features) ---
    # Values set to the median/mode or np.nan for imputation by the preprocessor
    user_input['normalized-losses'] = np.nan
    user_input['symboling'] = 0 # Integer default
    user_input['engine-location'] = 'front' 
    user_input['wheel-base'] = df_cleaned['wheel-base'].median()
    user_input['length'] = df_cleaned['length'].median()
    user_input['width'] = df_cleaned['width'].median()
    user_input['height'] = df_cleaned['height'].median()
    user_input['engine-type'] = df_cleaned['engine-type'].mode()[0]
    user_input['fuel-system'] = df_cleaned['fuel-system'].mode()[0]
    user_input['bore'] = df_cleaned['bore'].median()
    user_input['stroke'] = df_cleaned['stroke'].median()
    user_input['compression-ratio'] = df_cleaned['compression-ratio'].median()
    user_input['peak-rpm'] = df_cleaned['peak-rpm'].median()
    user_input['city-mpg'] = df_cleaned['city-mpg'].median()

# Prediction Button and Logic
with col_output:
    st.markdown("##### Predicted Price")
    if st.button("Predict Price", type="primary"):
        try:
            # 1. Create DataFrame in the exact feature order the model expects
            input_df = pd.DataFrame([user_input], columns=MODEL_FEATURES)
            
            # 2. CRITICAL FIX: Convert word numbers to digits for the numerical pipeline
            word_to_digit = {'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}
            
            for col in ['num-of-doors','num-of-cylinders']:
                # Map the string ('four') to the digit (4)
                input_df[col] = input_df[col].astype(str).str.lower().map(word_to_digit).fillna(input_df[col])
                # Ensure it's explicitly passed as float for the numerical pipeline (Imputer/Scaler)
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float) 

            # 3. CRITICAL FIX: Explicitly cast all categorical columns to 'object' dtype (string)
            # This prevents pandas from auto-casting them to numeric if they contain NaNs or only one value.
            for col in CATEGORICAL_FEATURES:
                 input_df[col] = input_df[col].astype('object')
            
            # The full pipeline handles all remaining preprocessing and prediction
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
            # Display a specific error pointing to the mismatch
            st.error(f"Prediction Error: Data structure mismatch. The column order is likely incorrect. Details: {e}")
    else:
        st.info("Adjust inputs and press **Predict Price**.")

# ==========================================================
# UI – MODEL INFO & CONCLUSION
# ==========================================================
st.divider()
st.header("📌 Model Performance and Conclusion")

col_model, col_conclusion = st.columns(2)

with col_model:
    st.subheader("Model Evaluation")
    
    # Metrics from your task.ipynb
    RMSE_VALUE = 2504.66
    R2_VALUE = 0.9009
    
    st.metric("Root Mean Squared Error (RMSE)", f"${RMSE_VALUE:,.2f}")
    st.metric("R-squared (R²)", f"{R2_VALUE:.4f}")

    st.markdown("""
    - **Model Used:** Random Forest Regressor.
    - **Preprocessing Pipeline:** Converts all categorical features into numerical vectors (One-Hot Encoding) and scales all numerical features (Standard Scaling).
    """)

with col_conclusion:
    st.subheader("Project Conclusion")
    st.markdown("""
    The Random Forest model achieved a high performance ($R^2 \approx 0.90$). The application successfully demonstrates the full machine learning workflow from EDA to interactive prediction.
    """)

plt.close('all')
