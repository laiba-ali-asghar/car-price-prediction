import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# Streamlit Page Config
# ==========================================================
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# ==========================================================
# Load Model & Data
# ==========================================================
@st.cache_resource
def load_pipeline_and_data():
    try:
        with open("car_price_model.pkl", "rb") as f:
            pipeline = pickle.load(f)
        df = pd.read_csv("carprice.csv", na_values="?")
        return pipeline, df
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

pipeline, df = load_pipeline_and_data()

# ==========================================================
# EDA Section
# ==========================================================
st.title("🚗 Car Price Prediction App")
st.header("Exploratory Data Analysis")

numeric_cols = df.select_dtypes(include=np.number).columns
fig, ax = plt.subplots(figsize=(10,6))
corr = df[numeric_cols].corr()
sns.heatmap(corr[['price']].sort_values(by='price', ascending=False), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.divider()

# ==========================================================
# User Input Section
# ==========================================================
st.header("🔮 Predict Car Price")

# Example defaults (you can expand to all features)
defaults = {
    'make': df['make'].mode()[0],
    'body-style': df['body-style'].mode()[0],
    'drive-wheels': df['drive-wheels'].mode()[0],
    'horsepower': int(df['horsepower'].median()),
    'engine-size': int(df['engine-size'].median()),
    'curb-weight': int(df['curb-weight'].median()),
    'highway-mpg': int(df['highway-mpg'].median()),
    'num-of-doors': 'four',
    'num-of-cylinders': 'four'
}

user_input = {}

# Categorical Inputs
user_input['make'] = st.selectbox("Make", df['make'].unique(), index=list(df['make'].unique()).index(defaults['make']))
user_input['body-style'] = st.selectbox("Body Style", df['body-style'].unique(), index=list(df['body-style'].unique()).index(defaults['body-style']))
user_input['drive-wheels'] = st.selectbox("Drive Wheels", df['drive-wheels'].unique(), index=list(df['drive-wheels'].unique()).index(defaults['drive-wheels']))
user_input['num-of-doors'] = st.selectbox("Doors", ['two','three','four','five'], index=2)
user_input['num-of-cylinders'] = st.selectbox("Cylinders", ['two','three','four','five','six','eight','twelve'], index=2)

# Numerical Inputs
user_input['horsepower'] = st.slider("Horsepower", 48, 288, defaults['horsepower'])
user_input['engine-size'] = st.slider("Engine Size", 61, 326, defaults['engine-size'])
user_input['curb-weight'] = st.slider("Curb Weight", 1488, 4066, defaults['curb-weight'])
user_input['highway-mpg'] = st.slider("Highway MPG", 16, 54, defaults['highway-mpg'])

# Fill missing columns with defaults for the pipeline
all_features = pipeline.named_steps['preprocessor'].feature_names_in_
for col in all_features:
    if col not in user_input:
        user_input[col] = np.nan  # preprocessor will handle missing values

# Predict Button
if st.button("Predict Price"):
    input_df = pd.DataFrame([user_input])
    
    # Convert word numbers to digits
    word_to_digit = {'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}
    for col in ['num-of-doors','num-of-cylinders']:
        input_df[col] = input_df[col].map(word_to_digit).astype(float)
    
    try:
        prediction = pipeline.predict(input_df)[0]
        st.success(f"Estimated Selling Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.markdown("**Note:** The Random Forest model uses preprocessing embedded in the pipeline. All missing values are handled automatically.")
