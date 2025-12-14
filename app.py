import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# EXPECTED FEATURES (MUST MATCH TRAINING PIPELINE)
# ==========================================================
ALL_FEATURES = [
    'symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height',
    'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio',
    'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg',
    'num-of-doors', 'num-of-cylinders',
    'make', 'fuel-type', 'aspiration', 'body-style',
    'drive-wheels', 'engine-location', 'engine-type', 'fuel-system'
]

# ==========================================================
# LOAD PIPELINE & DATA
# ==========================================================
@st.cache_resource
def load_pipeline_and_data():
    with open("car_price_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    df = pd.read_csv("carprice.csv", na_values="?")
    return pipeline, df

pipeline, df = load_pipeline_and_data()

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================
st.sidebar.title("🚘 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 EDA", "🔮 Prediction", "📌 Model Info"]
)

# ==========================================================
# HOME PAGE
# ==========================================================
if page == "🏠 Home":
    st.title("🚗 Car Price Prediction System")
    st.markdown("""
    ### 📌 Overview
    A **Machine Learning web application** that predicts used-car prices
    using a **Random Forest Regression Pipeline**.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "Random Forest")
    col2.metric("Dataset Size", df.shape[0])
    col3.metric("Total Features", len(ALL_FEATURES))

    st.image(
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70",
        use_container_width=True
    )

# ==========================================================
# EDA PAGE
# ==========================================================
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr[['price']].sort_values(by='price', ascending=False).head(10),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

    st.markdown("""
    **Key Insight:**  
    Engine Size, Horsepower, and Curb Weight are the most influential features.
    """)

# ==========================================================
# PREDICTION PAGE
# ==========================================================
elif page == "🔮 Prediction":
    st.title("🔮 Predict Car Price")

    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Make", df['make'].unique())
        body = st.selectbox("Body Style", df['body-style'].unique())
        drive = st.selectbox("Drive Wheels", df['drive-wheels'].unique())
        doors = st.selectbox("Doors", ['two', 'four'])
        cylinders = st.selectbox("Cylinders", ['four','five','six','eight'])

    with col2:
        horsepower = st.slider("Horsepower", 48, 288, int(df['horsepower'].median()))
        engine_size = st.slider("Engine Size", 61, 326, int(df['engine-size'].median()))
        curb_weight = st.slider("Curb Weight", 1488, 4066, int(df['curb-weight'].median()))
        highway_mpg = st.slider("Highway MPG", 16, 54, int(df['highway-mpg'].median()))

    # USER INPUT DICTIONARY
    user_input = {
        'make': make,
        'body-style': body,
        'drive-wheels': drive,
        'num-of-doors': doors,
        'num-of-cylinders': cylinders,
        'horsepower': horsepower,
        'engine-size': engine_size,
        'curb-weight': curb_weight,
        'highway-mpg': highway_mpg
    }

    # ENSURE ALL FEATURES EXIST
    for col in ALL_FEATURES:
        if col not in user_input:
            user_input[col] = np.nan

    if st.button("🚀 Predict Price"):
        input_df = pd.DataFrame([user_input])

        # Word → Digit conversion
        word_to_digit = {
            'two':2, 'three':3, 'four':4, 'five':5,
            'six':6, 'eight':8, 'twelve':12
        }
        for col in ['num-of-doors', 'num-of-cylinders']:
            input_df[col] = input_df[col].map(word_to_digit)

        prediction = pipeline.predict(input_df)[0]

        st.markdown(
            f"""
            <div style="background:#111827;padding:30px;border-radius:15px;text-align:center;">
                <h3 style="color:white;">Estimated Price</h3>
                <h1 style="color:#22c55e;">${prediction:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================================
# MODEL INFO PAGE
# ==========================================================
elif page == "📌 Model Info":
    st.title("📌 Model Details")

    st.markdown("""
    - **Algorithm:** Random Forest Regressor  
    - **Preprocessing:**  
      - Missing Value Imputation  
      - Feature Scaling  
      - One-Hot Encoding  
    """)

    col1, col2 = st.columns(2)
    col1.metric("RMSE", "$2504")
    col2.metric("R² Score", "0.90")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>© 2025 | Car Price Prediction App</p>",
    unsafe_allow_html=True
)
