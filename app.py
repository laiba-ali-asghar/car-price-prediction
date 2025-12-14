import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# ----------------------------------
# Page setup
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

    # Convert target explicitly
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Convert other numeric-looking columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

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
# Split features & target
# ----------------------------------
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

# ----------------------------------
# Train model
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# ----------------------------------
# Model score
# ----------------------------------
r2 = r2_score(y_test, model.predict(X_test))
st.write(f"### 📊 Model R² Score: **{r2:.2f}**")

# ----------------------------------
# User input sliders
# ----------------------------------
st.subheader("🔧 Enter Car Specifications")

user_input = {}
for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    default = float(X[col].mean())

    user_input[col] = st.slider(
        col,
        min_value=min_val,
        max_value=max_val,
        value=default
    )

input_df = pd.DataFrame([user_input])

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("💰 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Car Price: **${prediction:,.2f}**")

# ----------------------------------
# Dataset preview
# ----------------------------------
with st.expander("📄 Dataset Preview"):
    st.dataframe(df.head(10))
