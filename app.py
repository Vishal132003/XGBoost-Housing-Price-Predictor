import streamlit as st
import numpy as np
import pickle

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
        background: rgba(255,255,255,0.9);
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

# ---------------- Title ----------------
st.title("🏠 California Housing Price Prediction App")

st.write(
"""
This application predicts **house prices in California** using a trained
Machine Learning model.

Enter the property details in the sidebar and click **Predict Price**.
"""
)

st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.header("🏡 Property Details")

MedianIncome = st.sidebar.number_input(
    "Median Income of Area ($ per year)",
    min_value=10000,
    max_value=200000,
    value=60000,
    step=1000,
    help="Average yearly income of households in this area"
)

HouseAge = st.sidebar.slider(
    "Age of the House (Years)",
    min_value=1,
    max_value=60,
    value=20,
    help="How old the house/building is"
)

Rooms = st.sidebar.slider(
    "Total Rooms in the House",
    min_value=1,
    max_value=12,
    value=5,
    help="Total rooms including living room, kitchen, etc."
)

Bedrooms = st.sidebar.slider(
    "Number of Bedrooms",
    min_value=1,
    max_value=6,
    value=2,
    help="Total bedrooms in the house"
)

Population = st.sidebar.number_input(
    "Population of the Local Area",
    min_value=100,
    max_value=20000,
    value=3000,
    step=100,
    help="Approximate number of people living in the neighborhood"
)

Occupancy = st.sidebar.slider(
    "Average People per House",
    min_value=1,
    max_value=8,
    value=3,
    help="Average number of people living in each house"
)

Latitude = st.sidebar.number_input(
    "Latitude (Location Coordinate)",
    min_value=32.0,
    max_value=42.0,
    value=36.7,
    help="California latitude range approx 32–42"
)

Longitude = st.sidebar.number_input(
    "Longitude (Location Coordinate)",
    min_value=-125.0,
    max_value=-114.0,
    value=-119.4,
    help="California longitude range approx -125 to -114"
)

# ---------------- Contact Section ----------------
st.sidebar.markdown("---")
st.sidebar.header("📞 Contact")

st.sidebar.write("**Name:** Vishal Jadhav")
st.sidebar.write("**Email:** vaishnavijadhav01234@gmail.com")
st.sidebar.write("**Phone:** 8788965221")
st.sidebar.write("**LinkedIn:**")
st.sidebar.write("https://www.linkedin.com/in/vaishnavi-jadhav-465774327")

# ---------------- Convert Inputs ----------------
# Dataset expects income in units of $10,000
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
