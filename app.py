import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------------------------
# App Title
# ---------------------------------------
st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction App")

# ---------------------------------------
# Load Data
# ---------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("carprice.csv")
    df.replace("?", np.nan, inplace=True)
    df = df.dropna()
    return df

df = load_data()

# ---------------------------------------
# Select numeric features only
# ---------------------------------------
numeric_df = df.select_dtypes(include=[np.number])

X = numeric_df.drop("price", axis=1)
y = numeric_df["price"]

# ---------------------------------------
# Train Model
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------------------
# Model Performance
# ---------------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"R² Score: **{r2:.2f}**")

# ---------------------------------------
# User Input Section
# ---------------------------------------
st.subheader("🔧 Enter Car Specifications")

user_input = {}

for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())

    user_input[col] = st.slider(
        label=col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

input_df = pd.DataFrame([user_input])

# ---------------------------------------
# Prediction
# ---------------------------------------
if st.button("💰 Predict Car Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Car Price: **${prediction:,.2f}**")

# ---------------------------------------
# Show Raw Data (optional)
# ---------------------------------------
with st.expander("📄 View Dataset"):
    st.dataframe(df.head(20))
