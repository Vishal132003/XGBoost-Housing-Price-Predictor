import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- Load Dataset ----------------
data = fetch_california_housing()
X = data.data
y = data.target

# ---------------- Train Model ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor()
model.fit(X_train, y_train)

# ---------------- Title ----------------
st.title("🏠 California Housing Price Prediction")

st.write(
"""
This application predicts **California house prices** using a Machine Learning model.

Enter property details in the sidebar and click **Predict Price**.
"""
)

st.markdown("---")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🏡 Property Details")

MedianIncome = st.sidebar.number_input(
    "Median Income of Area ($ per year)",
    value=60000
)

HouseAge = st.sidebar.number_input(
    "House Age (Years)",
    value=20
)

Rooms = st.sidebar.number_input(
    "Total Rooms in the House",
    value=5
)

Bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    value=2
)

Population = st.sidebar.number_input(
    "Population of the Local Area",
    value=3000
)

Occupancy = st.sidebar.number_input(
    "Average People per House",
    value=3
)

Latitude = st.sidebar.number_input(
    "Latitude",
    value=36.7
)

Longitude = st.sidebar.number_input(
    "Longitude",
    value=-119.4
)

# ---------------- Contact ----------------
st.sidebar.markdown("---")
st.sidebar.header("📞 Contact")

st.sidebar.write("**Name:** Vishal Jadhav")
st.sidebar.write("**Email:** vishaljadhav132003@gmail.com")
st.sidebar.write("**Phone:** 8788965221")

# ---------------- Prepare Input ----------------
MedInc_model = MedianIncome / 10000

input_data = np.array([[
    MedInc_model,
    HouseAge,
    Rooms,
    Bedrooms,
    Population,
    Occupancy,
    Latitude,
    Longitude
]])

# ---------------- Prediction ----------------
st.subheader("Predict House Price")

if st.button("Predict Price"):

    prediction = model.predict(input_data)[0]
    price = prediction * 100000

    st.success(f"🏠 Estimated House Price: ${price:,.2f}")

# ---------------- Input Summary ----------------
st.markdown("---")
st.subheader("Input Summary")

st.write({
    "Median Income ($)": MedianIncome,
    "House Age (Years)": HouseAge,
    "Rooms": Rooms,
    "Bedrooms": Bedrooms,
    "Population": Population,
    "People per House": Occupancy,
    "Latitude": Latitude,
    "Longitude": Longitude
})
