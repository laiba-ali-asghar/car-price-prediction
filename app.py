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
    ["🏠 Home", "📊 Exploratory Data Analysis", "🔮 Price Prediction", "📌 Model Info"]
)

# ==========================================================
# HOME PAGE
# ==========================================================
if page == "🏠 Home":
    st.title("🚗 Car Price Prediction System")
    st.subheader("Machine Learning Powered Vehicle Price Estimation")

    st.markdown("""
    ### 📌 Project Overview
    This application predicts **car selling prices** using a  
    **Random Forest Regression model** trained on real automobile data.

    ### 🔧 Technologies Used
    - Python
    - Pandas, NumPy
    - Scikit-learn
    - Streamlit
    - Matplotlib & Seaborn

    ### 🎯 Objective
    Help users estimate a **fair market price** based on car specifications.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "Random Forest")
    col2.metric("Dataset Size", f"{df.shape[0]} Cars")
    col3.metric("Features Used", df.shape[1] - 1)

    st.image(
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70",
        use_container_width=True
    )

# ==========================================================
# EDA PAGE
# ==========================================================
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    st.markdown("### 🔍 Feature Correlation with Price")

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
    **Insight:**  
    - Engine Size  
    - Curb Weight  
    - Horsepower  

    have the **strongest influence** on car price.
    """)

    st.markdown("---")

    st.markdown("### 📈 Price Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['price'], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

# ==========================================================
# PREDICTION PAGE
# ==========================================================
elif page == "🔮 Price Prediction":
    st.title("🔮 Predict Car Price")
    st.markdown("Fill in the car details below:")

    defaults = {
        'make': df['make'].mode()[0],
        'body-style': df['body-style'].mode()[0],
        'drive-wheels': df['drive-wheels'].mode()[0],
        'horsepower': int(df['horsepower'].median()),
        'engine-size': int(df['engine-size'].median()),
        'curb-weight': int(df['curb-weight'].median()),
        'highway-mpg': int(df['highway-mpg'].median())
    }

    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Make", df['make'].unique())
        body = st.selectbox("Body Style", df['body-style'].unique())
        drive = st.selectbox("Drive Wheels", df['drive-wheels'].unique())
        doors = st.selectbox("Number of Doors", ['two', 'four'])
        cylinders = st.selectbox("Cylinders", ['four','five','six','eight'])

    with col2:
        horsepower = st.slider("Horsepower", 48, 288, defaults['horsepower'])
        engine_size = st.slider("Engine Size", 61, 326, defaults['engine-size'])
        curb_weight = st.slider("Curb Weight", 1488, 4066, defaults['curb-weight'])
        highway_mpg = st.slider("Highway MPG", 16, 54, defaults['highway-mpg'])

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

    # Fill missing columns
    for col in pipeline.named_steps['preprocessor'].feature_names_in_:
        if col not in user_input:
            user_input[col] = np.nan

    if st.button("🚀 Predict Price"):
        input_df = pd.DataFrame([user_input])

        word_to_digit = {
            'two':2, 'three':3, 'four':4, 'five':5,
            'six':6, 'eight':8, 'twelve':12
        }
        for col in ['num-of-doors', 'num-of-cylinders']:
            input_df[col] = input_df[col].map(word_to_digit)

        prediction = pipeline.predict(input_df)[0]

        st.markdown(
            f"""
            <div style="background:#1f2933;padding:30px;border-radius:15px;text-align:center;">
                <h3 style="color:white;">Estimated Car Price</h3>
                <h1 style="color:#22c55e;">${prediction:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================================
# MODEL INFO PAGE
# ==========================================================
elif page == "📌 Model Info":
    st.title("📌 Model Information")

    st.markdown("""
    ### 🧠 Model Details
    - **Algorithm:** Random Forest Regressor
    - **Preprocessing:**  
      - Missing Value Imputation  
      - Feature Scaling  
      - One-Hot Encoding
    """)

    col1, col2 = st.columns(2)
    col1.metric("RMSE", "$2504")
    col2.metric("R² Score", "0.90")

    st.markdown("""
    ### ✅ Conclusion
    The model performs strongly and generalizes well, making it suitable for
    real-world car price estimation.
    """)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>© 2025 | Car Price Prediction | Streamlit ML App</p>",
    unsafe_allow_html=True
)
