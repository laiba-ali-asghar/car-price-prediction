import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction")

# ----------------------------------
# Load & clean data
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("carprice.csv")
    df.replace("?", np.nan, inplace=True)

    # Explicit numeric conversion
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.dropna(inplace=True)
    return df

df = load_data()

# ----------------------------------
# Encode categorical columns
# ----------------------------------
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# ----------------------------------
# Split data
# ----------------------------------
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# Train model
# ----------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

r2 = r2_score(y_test, model.predict(X_test))
st.markdown(f"### 📊 Model R² Score: **{r2:.2f}**")

# ----------------------------------
# User inputs (SAFE)
# ----------------------------------
st.subheader("🔧 Enter Car Specifications")

user_input = {}

for col in X.columns:
    col_min = X[col].min()
    col_max = X[col].max()
    col_mean = X[col].mean()

    # Handle constant columns
    if col_min == col_max:
        user_input[col] = st.number_input(
            col,
            value=float(col_min),
            disabled=True
        )
        continue

    # Integer feature
    if np.issubdtype(X[col].dtype, np.integer):
        user_input[col] = st.slider(
            col,
            int(col_min),
            int(col_max),
            int(col_mean)
        )
    else:
        user_input[col] = st.slider(
            col,
            float(col_min),
            float(col_max),
            float(col_mean)
        )

input_df = pd.DataFrame([user_input])

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("💰 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Car Price: **${prediction:,.2f}**")

# ----------------------------------
# Data preview
# ----------------------------------
with st.expander("📄 Dataset Preview"):
    st.dataframe(df.head(10))
