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
# Feature Lists (must match pipeline training)
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
# Load Pipeline & Data
# ==========================================================
@st.cache_resource
def load_pipeline_and_data():
    try:
        # Load trained pipeline
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)

        # Load raw data for defaults/EDA
        df_raw = pd.read_csv(DATA_PATH, na_values="?")
        df = df_raw.dropna(subset=['price']).copy()

        # Ensure numeric columns are numeric
        num_like_cols = ['bore', 'stroke', 'horsepower', 'peak-rpm']
        for col in num_like_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return pipeline, df
    except FileNotFoundError:
        st.error(f"Required file not found: {MODEL_PATH} or {DATA_PATH}.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

pipeline, df_cleaned = load_pipeline_and_data()

# ==========================================================
# UI – INTRODUCTION & EDA
# ==========================================================
st.title("🚗 Car Price Prediction")
st.markdown("Predict a car's selling price using a trained Random Forest Regressor pipeline.")

st.divider()

# --- EDA Section ---
st.header("📊 Exploratory Data Analysis")
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
corr_matrix = df_cleaned[numeric_cols].corr()
sns.heatmap(corr_matrix[['price']].sort_values(by='price', ascending=False).head(8), 
            annot=True, fmt=".2f", cmap='coolwarm', cbar=False, ax=ax)
ax.set_title('Top Feature Correlation with Price')
st.pyplot(fig)
st.markdown("**Key Insight:** Engine size, curb weight, and horsepower correlate most with price.")
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

    # Categorical inputs
    col1, col2, col3 = st.columns(3)
    user_input['make'] = col1.selectbox("Make", df_cleaned['make'].unique(), 
                                        index=list(df_cleaned['make'].unique()).index(defaults['make']))
    user_input['body-style'] = col2.selectbox("Body Style", df_cleaned['body-style'].unique(), 
                                              index=list(df_cleaned['body-style'].unique()).index(defaults['body-style']))
    user_input['drive-wheels'] = col3.selectbox("Drive Wheels", df_cleaned['drive-wheels'].unique(), 
                                                index=list(df_cleaned['drive-wheels'].unique()).index(defaults['drive-wheels']))

    # Numerical sliders
    user_input['horsepower'] = st.slider("Horsepower (hp)", 48, 288, defaults['horsepower'])
    user_input['engine-size'] = st.slider("Engine Size (cc)", 61, 326, defaults['engine-size'])
    user_input['curb-weight'] = st.slider("Curb Weight (lbs)", 1488, 4066, defaults['curb-weight'])
    user_input['highway-mpg'] = st.slider("Highway MPG", 16, 54, defaults['highway-mpg'])

    # Other categorical features
    col4, col5, col6, col7 = st.columns(4)
    user_input['fuel-type'] = col4.radio("Fuel Type", df_cleaned['fuel-type'].unique(), 
                                         index=list(df_cleaned['fuel-type'].unique()).index(defaults['fuel-type']))
    user_input['aspiration'] = col5.radio("Aspiration", df_cleaned['aspiration'].unique(), 
                                          index=list(df_cleaned['aspiration'].unique()).index(defaults['aspiration']))
    user_input['num-of-doors'] = col6.selectbox("Doors (Word)", ['four', 'two'], index=0)
    user_input['num-of-cylinders'] = col7.selectbox("Cylinders (Word)", 
                                                    ['four', 'six', 'five', 'eight', 'two', 'three', 'twelve'], index=0)

    # Hidden features for pipeline
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

# --- Prediction ---
with col_output:
    st.markdown("##### Predicted Price")
    if st.button("Predict Price", type="primary"):
        try:
            # 1. Create DataFrame from user_input
            input_df = pd.DataFrame([user_input])

            # 2. Map word numbers to digits
            word_to_digit = {'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}
            for col in ['num-of-doors','num-of-cylinders']:
                input_df[col] = input_df[col].astype(str).str.lower().map(word_to_digit)

            # 3. Ensure numeric columns are float
            input_df[NUMERICAL_FEATURES] = input_df[NUMERICAL_FEATURES].astype(float)

            # 4. Ensure categorical columns are object
            input_df[CATEGORICAL_FEATURES] = input_df[CATEGORICAL_FEATURES].astype(object)

            # 5. Predict
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
            st.error(f"Prediction Error: {e}")
    else:
        st.info("Adjust inputs and press **Predict Price**.")

# ==========================================================
# Model Info & Conclusion
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
    - **Model:** Random Forest Regressor
    - **Pipeline:** One-Hot Encoding + Scaling
    """)

with col_conclusion:
    st.subheader("Project Conclusion")
    st.markdown("""
    Random Forest achieved high performance ($R^2 \approx 0.90$). 
    The app demonstrates a full ML workflow: EDA → Preprocessing → Prediction.
    """)

plt.close('all')
