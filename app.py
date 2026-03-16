import streamlit as st
import numpy as np
import pickle
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- Background Style ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(
            rgba(0,0,0,0.6),
            rgba(0,0,0,0.6)
        ),
        url("https://images.unsplash.com/photo-1560518883-ce09059eeffa");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        background: rgba(255,255,255,0.92);
        padding: 2rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Load or Train Model ----------------
model_path = "model_xgb.pkl"

if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except:
        st.warning("Model file corrupted. Retraining model...")
        data = fetch_california_housing()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        model = XGBRegressor()
        model.fit(X_train, y_train)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
else:
    st.warning("Model not found. Training new model...")

    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = XGBRegressor()
    model.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

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
    min_value=10000,
    max_value=200000,
    value=60000,
    step=1000
)

HouseAge = st.sidebar.slider(
    "House Age (Years)",
    1, 60, 20
)

Rooms = st.sidebar.slider(
    "Total Rooms in the House",
    1, 12, 5
)

Bedrooms = st.sidebar.slider(
    "Number of Bedrooms",
    1, 6, 2
)

Population = st.sidebar.number_input(
    "Population of the Local Area",
    100, 20000, 3000, 100
)

Occupancy = st.sidebar.slider(
    "Average People per House",
    1, 8, 3
)

Latitude = st.sidebar.number_input(
    "Latitude (Location Coordinate)",
    32.0, 42.0, 36.7
)

Longitude = st.sidebar.number_input(
    "Longitude (Location Coordinate)",
    -125.0, -114.0, -119.4
)

# ---------------- Contact Details ----------------
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
