import streamlit as st
import numpy as np
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- Background + Text Fix ----------------
st.markdown(
    """
    <style>

    .stApp {
        background-image: linear-gradient(
        rgba(0,0,0,0.75),
        rgba(0,0,0,0.75)),
        url("https://images.unsplash.com/photo-1560518883-ce09059eeffa");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        background-color: rgba(255,255,255,0.96);
        padding: 2rem;
        border-radius: 15px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    p, label, span {
        color: black !important;
        font-size:16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Load Model ----------------
try:
    with open("model_xgb.pkl", "rb") as file:
        model = pickle.load(file)
except:
    st.error("model_xgb.pkl not found or corrupted")
    st.stop()

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
    10000, 200000, 60000, 1000
)

HouseAge = st.sidebar.slider(
    "House Age (Years)",
    1, 60, 20
)

Rooms = st.sidebar.slider(
    "Total Rooms in House",
    1, 12, 5
)

Bedrooms = st.sidebar.slider(
    "Bedrooms",
    1, 6, 2
)

Population = st.sidebar.number_input(
    "Population of Area",
    100, 20000, 3000
)

Occupancy = st.sidebar.slider(
    "Average People per House",
    1, 8, 3
)

Latitude = st.sidebar.number_input(
    "Latitude",
    32.0, 42.0, 36.7
)

Longitude = st.sidebar.number_input(
    "Longitude",
    -125.0, -114.0, -119.4
)

# ---------------- Contact ----------------
st.sidebar.markdown("---")
st.sidebar.header("📞 Contact")

st.sidebar.write("Name: Vishal Jadhav")
st.sidebar.write("Email: vishaljadhav132003@gmail.com")
st.sidebar.write("Phone: 9529935831")

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
