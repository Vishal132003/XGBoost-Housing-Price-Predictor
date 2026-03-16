import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import fetch_california_housing

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load Model
try:
    with open("model_xgb.pkl", "rb") as file:
        model = pickle.load(file)
except:
    st.error("model_xgb.pkl file not found")
    st.stop()

# Dataset
housing = fetch_california_housing()

# Title
st.title("🏠 California Housing Price Prediction App")
st.write("Machine Learning app to predict house prices using XGBoost model")

st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.header("Enter Housing Details")

MedInc = st.sidebar.slider("Median Income", 0.0, 15.0, 5.0)
HouseAge = st.sidebar.slider("House Age", 1, 60, 20)
AveRooms = st.sidebar.slider("Average Rooms", 1.0, 15.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("Population", 100, 5000, 1000)
AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -120.0)

st.sidebar.markdown("---")

# Contact Details
st.sidebar.header("📞 Contact")
st.sidebar.write("**Name:** Vishal Jadhav")
st.sidebar.write("**Email:** vaishnavijadhav01234@gmail.com")
st.sidebar.write("**Phone:** 8788965221")
st.sidebar.write("**LinkedIn:**")
st.sidebar.write("https://www.linkedin.com/in/vaishnavi-jadhav-465774327")

st.sidebar.markdown("---")
st.sidebar.info("This project predicts California housing prices using Machine Learning.")

# ---------------- Prediction ----------------

input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude]])

st.subheader("Predict House Price")

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    price = prediction * 100000

    st.success(f"🏡 Estimated House Price: ${price:,.2f}")

# ---------------- Input Summary ----------------

st.subheader("Input Summary")

st.write({
    "Median Income": MedInc,
    "House Age": HouseAge,
    "Average Rooms": AveRooms,
    "Average Bedrooms": AveBedrms,
    "Population": Population,
    "Average Occupancy": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude
})
