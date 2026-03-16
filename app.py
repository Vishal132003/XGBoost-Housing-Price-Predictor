import streamlit as st
import numpy as np
import pickle

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

# ---------------- Load Model ----------------
try:
    file = open("model_xgb.pkl", "rb")
    model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ model_xgb.pkl file not found. Please place it in the same folder as app.py")
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
    min_value=10000,
    max_value=200000,
    value=60000,
    step=1000,
    help="Average yearly income of households in this area"
)

HouseAge = st.sidebar.slider(
    "House Age (Years)",
    1, 60, 20,
    help="Age of the house"
)

Rooms = st.sidebar.slider(
    "Total Rooms in the House",
    1, 12, 5,
    help="Total rooms including living room, kitchen etc."
)

Bedrooms = st.sidebar.slider(
    "Number of Bedrooms",
    1, 6, 2,
    help="Total bedrooms in the house"
)

Population = st.sidebar.number_input(
    "Population of the Local Area",
    100, 20000, 3000, 100,
    help="Approximate population of the neighborhood"
)

Occupancy = st.sidebar.slider(
    "Average People per House",
    1, 8, 3,
    help="Average number of people living in each house"
)

Latitude = st.sidebar.number_input(
    "Latitude (Location Coordinate)",
    32.0, 42.0, 36.7,
    help="California latitude range approx 32–42"
)

Longitude = st.sidebar.number_input(
    "Longitude (Location Coordinate)",
    -125.0, -114.0, -119.4,
    help="California longitude range approx -125 to -114"
)

# ---------------- Contact Details ----------------
st.sidebar.markdown("---")
st.sidebar.header("📞 Contact")

st.sidebar.write("**Name:** Vishal Jadhav")
st.sidebar.write("**Email:** vishaljadhav132003@gmail.com")
st.sidebar.write("**Phone:** 8788965221")

# ---------------- Convert Inputs ----------------
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
