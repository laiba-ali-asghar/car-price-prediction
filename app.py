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
# Feature Lists (MUST match your ColumnTransformer setup in task.ipynb)
# ==========================================================

def get_feature_lists():
    """Defines the feature lists required for type enforcement and UI generation."""
    # Features treated as Numerical by the preprocessor (for Scaling/Imputation)
    NUMERICAL_FEATURES = [
        'symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 
        'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
        'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg'
    ]
    # Features treated as Categorical by the preprocessor (for OneHotEncoding)
    CATEGORICAL_FEATURES = [
        'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 
        'fuel-system'
    ]
    return NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# ==========================================================
# Helper Functions (Cached)
# ==========================================================

@st.cache_data
def clean_data(df):
    """Applies necessary cleaning steps for both EDA and prediction input."""
    df = df.copy()
    
    # Drop rows where the target variable 'price' is missing for EDA/Defaults
    if 'price' in df.columns:
        df = df.dropna(subset=['price']) 

    # Handle the text-to-number columns like 'num-of-doors' as strings for the preprocessor
    # The clean data is primarily used to get consistent unique categories for UI widgets.
    
    # We explicitly convert numeric columns to numeric types for EDA/UI defaults
    num_like_cols = ['bore', 'stroke', 'horsepower', 'peak-rpm']
    for col in num_like_cols:
        if col in df.columns:
             # Errors='coerce' will turn '?' or other non-numeric strings into NaN
             df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

@st.cache_resource
def load_assets():
    """Load model and preprocessor."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = pickle.load(f)
        df_raw = pd.read_csv(DATA_PATH, na_values='?')
        df_cleaned = clean_data(df_raw)
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
# Prediction Function (FIXED FOR TYPE INCONSISTENCY)
# ==========================================================
def predict_price(input_data):
    """Takes user input, preprocesses it, and returns a price prediction."""
    
    # 1. Convert input dict to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 2. Type Enforcement FIX (CRITICAL)
    # Ensure all numerical columns are float
    for col in NUMERICAL_FEATURES:
        if col in input_df.columns:
            # Use pd.to_numeric with 'coerce' to handle NaNs/objects gracefully, then ensure float type
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float) 

    # Ensure all categorical columns are string/object
    for col in CATEGORICAL_FEATURES:
        if col in input_df.columns:
            # Cast all categorical inputs to string to prevent object-to-numeric issues
            input_df[col] = input_df[col].astype(str)

    # 3. Preprocess and Predict
    try:
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}. Ensure the number and order of features in the input match the preprocessor.")
        return np.nan


# ==========================================================
# UI – INTRODUCTION & EDA (Single Page)
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
numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
corr_matrix = df_cleaned[numeric_cols].corr()
sns.heatmap(corr_matrix[['price']].sort_values(by='price', ascending=False), 
            annot=True, fmt=".2f", cmap='coolwarm', cbar=False, ax=ax)
ax.set_title('Top Feature Correlation with Price')
st.pyplot(fig)

st.markdown("""
**Key Insight:** The heatmap highlights the strongest positive predictors of price: 
**Engine Size (0.87)**, **Curb Weight (0.84)**, and **Horsepower (0.81)**. These features are the most important for the model.
""")

st.subheader("2. Summary Statistics")
st.dataframe(df_cleaned[['price', 'horsepower', 'engine-size', 'curb-weight']].describe().T)

st.divider()

# ==========================================================
# UI – PREDICTION INTERFACE
# ==========================================================
st.header("🔮 Interactive Price Prediction")
st.markdown("Select features below to get a real-time price prediction.")

# --- UI Defaults ---
# Default Imputed/Median Values (must match what the model was trained with)
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
    
    # Row 3: Other Categorical Features
    col4, col5, col6, col7 = st.columns(4)
    user_input['fuel-type'] = col4.radio("Fuel Type", df_cleaned['fuel-type'].unique(), index=list(df_cleaned['fuel-type'].unique()).index(defaults['fuel-type']))
    user_input['aspiration'] = col5.radio("Aspiration", df_cleaned['aspiration'].unique(), index=list(df_cleaned['aspiration'].unique()).index(defaults['aspiration']))
    # Note: These MUST be in text format for the categorical encoder
    user_input['num-of-doors'] = col6.selectbox("Doors", ['four', 'two'], index=0)
    user_input['num-of-cylinders'] = col7.selectbox("Cylinders", ['four', 'six', 'five', 'eight', 'two', 'three', 'twelve'], index=0)

    # --- Hidden Imputed Values (Needed to complete the 25 features of the preprocessor) ---
    user_input['normalized-losses'] = np.nan # Will be imputed by the preprocessor
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
    
    # These values are taken from the task.ipynb output and should be confirmed
    RMSE_VALUE = 2504.66
    R2_VALUE = 0.9009
    
    st.metric("Root Mean Squared Error (RMSE)", f"${RMSE_VALUE:,.2f}")
    st.metric("R-squared (R²)", f"{R2_VALUE:.4f}")

    st.markdown("""
    - **Model Used:** Random Forest Regressor (Regression Model).
    - **Preprocessing Pipeline:** Imputation for missing values, One-Hot Encoding for categorical features, and Standard Scaling for numerical features.
    """)

with col_conclusion:
    st.subheader("Project Conclusion")
    st.markdown("""
    This project successfully achieved its goal by building a highly accurate price prediction model ($R^2 \approx 0.90$). The deployment to Streamlit Cloud provides a crucial interactive component.

    **Final Takeaway:** Maintaining strict data type consistency between the training environment (Jupyter Notebook) and the deployment environment (Streamlit) is the most critical factor for successfully using saved scikit-learn preprocessing pipelines.
    """)

# Ensure plots are closed
plt.close('all')
