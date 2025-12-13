import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================================
# Streamlit Page Configuration
# ==========================================================
st.set_page_config(page_title="Car Price Prediction Project", layout="wide")

# ==========================================================
# File Paths
# ==========================================================
MODEL_PATH = "car_price_model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"
DATA_PATH = "carprice.csv"

# ==========================================================
# Feature Lists (CRITICAL - UPDATED for the final fix)
# ==========================================================

def get_feature_lists():
    """Defines the feature lists matching the ColumnTransformer setup in task.ipynb."""
    # NOTE: 'num-of-doors' and 'num-of-cylinders' MUST be NUMERICAL because 
    # the training script converted them from words to digits (e.g., 'four' -> 4)
    NUMERICAL_FEATURES = [
        'symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 
        'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
        'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg',
        'num-of-doors', 'num-of-cylinders' # <--- ADDED THESE TWO HERE
    ]
    # Features treated as Categorical by the preprocessor (all other objects)
    CATEGORICAL_FEATURES = [
        'make', 'fuel-type', 'aspiration', 'body-style', 
        'drive-wheels', 'engine-location', 'engine-type', 'fuel-system'
    ]
    return NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# ==========================================================
# Helper Functions (Cached)
# ==========================================================

@st.cache_data
def clean_data(df):
    """
    Applies necessary cleaning steps for both EDA and prediction input.
    CRITICAL: Converts word-based numbers to digits for features treated as NUMERICAL.
    """
    df = df.copy()
    
    # Drop rows where the target variable 'price' is missing for EDA/Defaults
    if 'price' in df.columns:
        df = df.dropna(subset=['price']) 

    # Dictionary for converting word-based numbers to digits
    word_to_digit = {
        'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
        'eight': 8, 'twelve': 12
    }

    cols_to_convert = ['num-of-doors', 'num-of-cylinders']
    for col in cols_to_convert:
        if col in df.columns:
            # Map word to digit, and fillna with original value 
            df[col] = df[col].astype(str).str.lower().map(word_to_digit).fillna(df[col])
            # Now convert the cleaned column to numeric (this is what the preprocessor expects!)
            df[col] = pd.to_numeric(df[col], errors='coerce')


    # Explicitly convert other known numerical-like columns for EDA/UI defaults
    num_like_cols = ['bore', 'stroke', 'horsepower', 'peak-rpm']
    for col in num_like_cols:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

@st.cache_resource
def load_assets():
    """Load model, preprocessor, and data."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = pickle.load(f)
        df_raw = pd.read_csv(DATA_PATH, na_values='?')
        df_cleaned = clean_data(df_raw) # This instance of clean_data handles the data for UI defaults
        return model, preprocessor, df_cleaned
    except FileNotFoundError:
        st.error("Required project files (model, preprocessor, or data) not found. Check GitHub repository.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during asset loading: {e}")
        st.stop()


model, preprocessor, df_cleaned = load_assets()
NUMERICAL_FEATURES, CATEGORICAL_FEATURES = get_feature_lists()


# ==========================================================
# Prediction Function (FINAL FIXED VERSION)
# ==========================================================
def predict_price(input_data):
    """Takes user input, preprocesses it, and returns a price prediction."""
    
    # 1. Convert input dict to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 2. Convert text numbers to digits (CRITICAL FIX)
    # We call clean_data() which now contains the logic to convert 'four' -> 4.0
    input_df = clean_data(input_df) 
    
    # 3. Type Enforcement FIX: Ensure all features have the exact type the preprocessor expects.
    
    # Ensure all numerical columns are float (including num-of-doors/cylinders)
    for col in NUMERICAL_FEATURES:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float) 

    # Ensure all categorical columns are string/object
    for col in CATEGORICAL_FEATURES:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

    # 4. Preprocess and Predict
    try:
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction failed. Final error: {e}. Check feature lists.")
        return np.nan


# ==========================================================
# UI – INTRODUCTION & EDA (Minimalist Single Page)
# ==========================================================
st.title("🚗 Data Science Capstone: Car Price Prediction")
st.markdown("""
This project applies the concepts of **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, 
and **Machine Learning (Random Forest Regressor)** to predict a car's selling price based on its features.
""")

st.divider()

# --- EDA Section ---
st.header("📊 Exploratory Data Analysis & Insights")
st.subheader("1. Feature Correlation with Price")

# Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
# Ensure we use the cleaned numerical columns
numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
corr_matrix = df_cleaned[numeric_cols].corr()
sns.heatmap(corr_matrix[['price']].sort_values(by='price', ascending=False).head(8), 
            annot=True, fmt=".2f", cmap='coolwarm', cbar=False, ax=ax)
ax.set_title('Top Feature Correlation with Price')
st.pyplot(fig)

st.markdown("""
**Key Insight:** The heatmap confirms that the most influential features are those related to **size and engine power** (Engine Size, Curb Weight, Horsepower) with strong positive correlations.
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
    'horsepower': df_cleaned['horsepower'].median(),
    'engine-size': df_cleaned['engine-size'].median(),
    'curb-weight': df_cleaned['curb-weight'].median(),
    'highway-mpg': df_cleaned['highway-mpg'].median()
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
    user_input['horsepower'] = st.slider("Horsepower (hp)", 48, 288, int(defaults['horsepower']))
    user_input['engine-size'] = st.slider("Engine Size (cc)", 61, 326, int(defaults['engine-size']))
    user_input['curb-weight'] = st.slider("Curb Weight (lbs)", 1488, 4066, int(defaults['curb-weight']))
    user_input['highway-mpg'] = st.slider("Highway MPG", 16, 54, int(defaults['highway-mpg']))
    
    # Row 3: Other Categorical Features (Passed as strings, converted to numbers in clean_data)
    col4, col5, col6, col7 = st.columns(4)
    user_input['fuel-type'] = col4.radio("Fuel Type", df_cleaned['fuel-type'].unique(), index=list(df_cleaned['fuel-type'].unique()).index(defaults['fuel-type']))
    user_input['aspiration'] = col5.radio("Aspiration", df_cleaned['aspiration'].unique(), index=list(df_cleaned['aspiration'].unique()).index(defaults['aspiration']))
    # These must remain as strings 'four'/'two' in the input_data to be mapped to numbers later
    user_input['num-of-doors'] = col6.selectbox("Doors", ['four', 'two'], index=0)
    user_input['num-of-cylinders'] = col7.selectbox("Cylinders", ['four', 'six', 'five', 'eight', 'two', 'three', 'twelve'], index=0)

    # --- Hidden Imputed Values (Needed for the preprocessor) ---
    user_input['normalized-losses'] = np.nan
    user_input['symboling'] = 0
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


with col_output:
    st.markdown("##### Predicted Price")
    if st.button("Predict Price", type="primary"):
        predicted_price = predict_price(user_input)
        if not np.isnan(predicted_price):
            st.markdown(
                f"""
                <div style="background-color:#1f4e79;padding:25px;border-radius:12px;text-align:center;">
                    <h3 style="color:white;">Estimated Selling Price</h3>
                    <h1 style="color:#f9c74f;">${predicted_price:,.2f}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
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
    
    RMSE_VALUE = 2504.66
    R2_VALUE = 0.9009
    
    st.metric("Root Mean Squared Error (RMSE)", f"${RMSE_VALUE:,.2f}")
    st.metric("R-squared (R²)", f"{R2_VALUE:.4f}")

    st.markdown("""
    - **Model Used:** Random Forest Regressor.
    - **Preprocessing Pipeline:** Converts all categorical features into numerical vectors (One-Hot Encoding) and scales all numerical features (Standard Scaling) for optimal model training.
    """)

with col_conclusion:
    st.subheader("Project Conclusion")
    st.markdown("""
    The Random Forest model achieved a high performance ($R^2 \approx 0.90$). 

    **Key Takeaway:** The success of the prediction relies entirely on replicating the exact data cleaning and transformation steps (especially converting 'four' to $4.0$) that were performed during model training.
    """)

plt.close('all')
