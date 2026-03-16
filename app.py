import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- Background Image ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1560518883-ce09059eeffa");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        background: rgba(255,255,255,0.85);
        padding: 2rem;
        border-radius: 10px;
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
    st.error("model_xgb.pkl file not found")
    st.stop()

# ---------------- Dataset ----------------
housing = fetch_california_housing()

# ---------------- Title ----------------
st.title("🏠 California Housing Price Prediction")
st.write("Predict California house prices using a Machine Learning model (XGBoost).")

st.markdown("---")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🏡 Housing Details")

MedInc = st.sidebar.slider(
    "Median Income (× $10,000)",
    0.5, 15.0, 5.0, 0.1
)

HouseAge = st.sidebar.slider(
    "House Age (Years)",
    1, 52, 25
)

AveRooms = st.sidebar.slider(
    "Average Rooms per House",
    2.0, 10.0, 5.0, 0.1
)

AveBedrms = st.sidebar.slider(
    "Average Bedrooms per House",
    0.5, 5.0, 1.0, 0.1
)

Population = st.sidebar.slider(
    "Population in Block",
    100, 5000, 1000, 50
)

AveOccup = st.sidebar.slider(
    "Average Occupancy (People per House)",
    1.0, 6.0, 3.0, 0.1
)

Latitude = st.sidebar.slider(
    "Latitude (°)",
    32.5, 42.0, 36.5, 0.1
)

Longitude = st.sidebar.slider(
    "Longitude (°)",
    -124.5, -114.0, -119.5, 0.1
)

st.sidebar.markdown("---")

# ---------------- Contact Details ----------------
st.sidebar.header("📞 Contact")

st.sidebar.write("**Name:** Vishal Jadhav")
st.sidebar.write("**Email:** vaishnavijadhav01234@gmail.com")
st.sidebar.write("**Phone:** 8788965221")

st.sidebar.write("**LinkedIn:**")
st.sidebar.write("https://www.linkedin.com/in/vaishnavi-jadhav-465774327")

st.sidebar.markdown("---")
st.sidebar.info("Machine Learning project built using Streamlit and XGBoost.")

# ---------------- Prediction ----------------
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude]])

st.subheader("Predict House Price")

if st.button("Predict Price"):

    prediction = model.predict(input_data)[0]
    price = prediction * 100000

    st.success(f"🏠 Estimated House Price: ${price:,.2f}")

# ---------------- Input Summary ----------------
st.markdown("---")
st.subheader("Input Summary")

st.write({
    "Median Income ($)": MedInc * 10000,
    "House Age (Years)": HouseAge,
    "Average Rooms": AveRooms,
    "Average Bedrooms": AveBedrms,
    "Population": Population,
    "Average Occupancy": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude
})
