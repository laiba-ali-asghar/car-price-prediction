import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt 

# ==========================================================
# Streamlit Page Configuration
# ==========================================================
st.set_page_config(page_title="Car Price Prediction Project", layout="wide")

# ==========================================================
# File Paths (MUST match your saved files)
# ==========================================================
MODEL_PATH = "car_price_model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"
DATA_PATH = "carprice.csv"

# ==========================================================
# Helper Functions
# ==========================================================

# 1. CLEANING FUNCTION (MUST match cleaning logic in task.ipynb)
@st.cache_data
def clean_data(df):
    """Applies necessary cleaning steps for both EDA and prediction."""
    # Handle NaN in target variable for training/EDA data (not strictly needed for single prediction, but good practice)
    if 'price' in df.columns:
        df = df.dropna(subset=['price']).copy()
    else:
        df = df.copy()

    # Dictionary for converting word-based numbers to digits
    word_to_digit = {
        'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
        'eight': 8, 'twelve': 12
    }

    # Apply conversion to specified columns (as strings for mapping)
    cols_to_convert = ['num-of-doors', 'num-of-cylinders']
    for col in cols_to_convert:
        if col in df.columns:
            # Map word to digit, and fillna with original value (text if not mapped)
            df[col] = df[col].astype(str).str.lower().map(word_to_digit).fillna(df[col])
            # Now convert the cleaned column back to its original text representation 
            # (which the ColumnTransformer expects for OneHotEncoding)
            df[col] = df[col].astype(str)
            
    return df

# 2. NUMERICAL FEATURES LIST (CRITICAL for fixing the type error)
def get_numerical_features():
    """Returns the list of numerical features used in the training script."""
    # This list must be identical to the one used in your ColumnTransformer in task.ipynb
    return [
        'symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 
        'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
        'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg'
    ]

# ==========================================================
# Load Model, Preprocessor & Raw Data (cached)
# ==========================================================
@st.cache_resource
def load_assets():
    """Load model and preprocessor (critical assets)."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except FileNotFoundError:
        st.error("Required model or preprocessor files not found. Ensure 'car_price_model.pkl' and 'preprocessor.pkl' exist.")
        return None, None

@st.cache_data
def load_raw_data():
    """Load and clean the raw data for EDA and category extraction."""
    try:
        # Load raw data with '?' as NaN
        df_raw = pd.read_csv(DATA_PATH, na_values='?')
        # Clean and get a stable DataFrame for EDA
        df_cleaned = clean_data(df_raw) 
        return df_cleaned
    except FileNotFoundError:
        st.error("Raw data file 'carprice.csv' not found.")
        return pd.DataFrame()


model, preprocessor = load_assets()
df_cleaned = load_raw_data()

if model is None or preprocessor is None or df_cleaned.empty:
    st.stop()


# ==========================================================
# Prediction Function (FIXED)
# ==========================================================
def predict_price(input_data):
    """Takes user input, preprocesses it, and returns a price prediction."""
    # 1. Convert input dict to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 2. Apply the same data cleaning steps (text numbers to original text representation)
    input_df = clean_data(input_df) 
    
    # 3. CRITICAL FIX: Explicitly cast all numerical columns to float
    num_features = get_numerical_features()
    for col in num_features:
        if col in input_df.columns:
            # Coerce non-numeric values (like the '?' from user-provided NaNs) to NaN, then ensure float type
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float) 

    # 4. Preprocess the input data using the fitted preprocessor
    input_processed = preprocessor.transform(input_df)
    
    # 5. Make prediction
    prediction = model.predict(input_processed)[0]
    
    return prediction


# ==========================================================
# UI – INTRODUCTION
# ==========================================================
st.title("🚗 Car Price Prediction Project")
st.markdown("""
This is the final project submission for the Introduction to Data Science course. 
The objective is to analyze a comprehensive car dataset and build a machine learning model 
to accurately predict car prices based on various features.
""")

# ==========================================================
# UI – EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================
st.divider()
st.header("📊 Exploratory Data Analysis (EDA)")
st.markdown(f"The analysis is performed on the **{DATA_PATH}** dataset, consisting of **{df_cleaned.shape[0]}** clean records and **{df_cleaned.shape[1]}** features.")

tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Correlation Heatmap", "Interactive Visualizations"])

with tab1:
    st.subheader("1. Summary Statistics & Missing Values")
    col_desc, col_miss = st.columns(2)
    
    with col_desc:
        st.markdown("##### Descriptive Statistics (Numerical Features)")
        st.dataframe(df_cleaned.select_dtypes(include=np.number).describe().T)
    
    with col_miss:
        st.markdown("##### Missing Value Analysis")
        missing_data = df_cleaned.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        missing_df = pd.DataFrame({'Feature': missing_data.index, 'Missing Count': missing_data.values})
        st.dataframe(missing_df, hide_index=True)


with tab2:
    st.subheader("2. Feature Correlation Analysis")
    st.markdown("A heatmap showing the linear correlation between all numerical features. **Engine Size**, **Curb Weight**, and **Horsepower** show the highest positive correlation with **Price**.")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    # Select only numerical features for correlation
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
    corr_matrix = df_cleaned[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Numerical Features')
    st.pyplot(fig)


with tab3:
    st.subheader("3. Interactive Feature Distribution")
    
    # Get lists of feature names for selection
    numerical_features_eda = [col for col in df_cleaned.select_dtypes(include=np.number).columns.tolist() if col != 'price']
    categorical_features = df_cleaned.select_dtypes(include='object').columns.tolist()

    viz_type = st.radio("Choose Visualization Type:", ["Distribution (Histogram)", "Price vs. Category (Box Plot)"], horizontal=True)

    if viz_type == "Distribution (Histogram)":
        feature = st.selectbox("Select Numerical Feature to Visualize:", numerical_features_eda)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_cleaned[feature].dropna(), kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature.replace("-", " ").title()}')
        ax.set_xlabel(feature.replace("-", " ").title())
        st.pyplot(fig)

    elif viz_type == "Price vs. Category (Box Plot)":
        feature = st.selectbox("Select Categorical Feature for Price Analysis:", categorical_features)
        
        # Order by median price for better visualization
        order = df_cleaned.groupby(feature)['price'].median().sort_values(ascending=False).index
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=feature, y='price', data=df_cleaned, order=order, ax=ax)
        ax.set_title(f'Price Distribution by {feature.replace("-", " ").title()}')
        ax.set_xlabel(feature.replace("-", " ").title())
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)


# ==========================================================
# UI – PREDICTION INTERFACE
# ==========================================================
st.divider()
st.header("🔮 Price Prediction Interface")
st.markdown("Use the sliders and selectors below to specify the features of the car and predict its selling price.")

# Get lists for the selectors based on the cleaned data
categorical_defaults = {
    'make': df_cleaned['make'].mode()[0],
    'fuel-type': df_cleaned['fuel-type'].mode()[0],
    'aspiration': df_cleaned['aspiration'].mode()[0],
    'body-style': df_cleaned['body-style'].mode()[0],
    'drive-wheels': df_cleaned['drive-wheels'].mode()[0],
}
numerical_ranges = {
    'horsepower': (48, 288, df_cleaned['horsepower'].median()),
    'engine-size': (61, 326, df_cleaned['engine-size'].median()),
    'curb-weight': (1488, 4066, df_cleaned['curb-weight'].median()),
    'highway-mpg': (16, 54, df_cleaned['highway-mpg'].median())
}


# Collect user input
user_input = {}
col1, col2 = st.columns(2)

with col1:
    st.subheader("Car Features")
    
    # Categorical Inputs
    user_input['make'] = st.selectbox("Make", df_cleaned['make'].unique(), index=list(df_cleaned['make'].unique()).index(categorical_defaults['make']))
    user_input['body-style'] = st.selectbox("Body Style", df_cleaned['body-style'].unique(), index=list(df_cleaned['body-style'].unique()).index(categorical_defaults['body-style']))
    user_input['drive-wheels'] = st.selectbox("Drive Wheels", df_cleaned['drive-wheels'].unique(), index=list(df_cleaned['drive-wheels'].unique()).index(categorical_defaults['drive-wheels']))
    user_input['fuel-type'] = st.radio("Fuel Type", df_cleaned['fuel-type'].unique(), index=list(df_cleaned['fuel-type'].unique()).index(categorical_defaults['fuel-type']), horizontal=True)
    user_input['aspiration'] = st.radio("Aspiration", df_cleaned['aspiration'].unique(), index=list(df_cleaned['aspiration'].unique()).index(categorical_defaults['aspiration']), horizontal=True)
    
    # Numerical-like inputs that are passed as the text format the model expects
    user_input['num-of-doors'] = st.selectbox("Number of Doors", ['four', 'two'], index=0)
    user_input['num-of-cylinders'] = st.selectbox("Number of Cylinders", ['four', 'six', 'five', 'eight', 'two', 'three', 'twelve'], index=0)

    # All necessary features for the model must be present, even if imputed later
    user_input['normalized-losses'] = np.nan
    user_input['symboling'] = 0
    user_input['engine-location'] = 'front' # Mode
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


    # Numerical Slider Inputs
    user_input['horsepower'] = st.slider("Horsepower (hp)", numerical_ranges['horsepower'][0], numerical_ranges['horsepower'][1], int(numerical_ranges['horsepower'][2]))
    user_input['engine-size'] = st.slider("Engine Size (cc)", numerical_ranges['engine-size'][0], numerical_ranges['engine-size'][1], int(numerical_ranges['engine-size'][2]))
    user_input['curb-weight'] = st.slider("Curb Weight (lbs)", numerical_ranges['curb-weight'][0], numerical_ranges['curb-weight'][1], int(numerical_ranges['curb-weight'][2]))
    user_input['highway-mpg'] = st.slider("Highway MPG", numerical_ranges['highway-mpg'][0], numerical_ranges['highway-mpg'][1], int(numerical_ranges['highway-mpg'][2]))


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
st.header("📌 Model Information and Performance")
st.markdown("""
The model used is a **Random Forest Regressor**, which is well-suited for non-linear regression problems like price prediction.
""")

# Display the evaluation metrics from task.ipynb (Using placeholders from previous successful runs)
st.subheader("Model Evaluation on Test Set")
col_rmse, col_r2 = st.columns(2)

RMSE_VALUE = 2504.66
R2_VALUE = 0.9009

col_rmse.metric("Root Mean Squared Error (RMSE)", f"${RMSE_VALUE:,.2f}")
col_r2.metric("R-squared (R²)", f"{R2_VALUE:.4f}", help="Indicates that the model explains approximately 90% of the variance in car prices.")

st.markdown("""
- **Model:** Random Forest Regressor (Ensemble model providing robust prediction).
- **Preprocessing:** All numerical features were **Standard Scaled** and all categorical features were converted using **One-Hot Encoding**. Missing values were handled via **Imputation** (mean for numerical, mode for categorical).
""")


# ==========================================================
# UI – CONCLUSION
# ==========================================================
st.divider()
st.header("✅ Conclusion")
st.markdown("""
This project successfully developed a car price prediction model with high performance ($R^2 \\approx 0.90$). 

**Key Takeaways:**
* **Data Preparation is Paramount:** Correctly handling mixed data types and ensuring consistent input features in the Streamlit app was the key to successful deployment and accurate predictions.
* **Engine Metrics Dominate:** The EDA confirmed that features related to the engine and physical size (`horsepower`, `engine-size`, `curb-weight`) are the strongest predictors of a car's price.
* **Model Implementation:** By saving and loading the **`preprocessor`** alongside the trained **`model`**, we ensured that the Streamlit application could transform raw user input exactly as the training data was transformed, allowing for accurate, real-time predictions.
""")

# Ensure plots are closed
plt.close('all')
