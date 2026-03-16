import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="California Housing AI Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(120deg,#1e3c72,#2a5298);
color:white;
}

.block-container{
background: rgba(255,255,255,0.1);
padding:30px;
border-radius:15px;
backdrop-filter: blur(10px);
}

h1,h2,h3{
color:white;
}

.stButton>button{
background-color:#ff4b4b;
color:white;
font-size:18px;
border-radius:10px;
height:3em;
width:200px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
try:
    model = pickle.load(open("model_xgb.pkl","rb"))
except:
    st.error("❌ model_xgb.pkl file not found. Please keep it in the same folder.")
    st.stop()

# ---------------- Title ----------------
st.title("🏠 California Housing Price Predictor")

st.write("""
This AI application predicts **California house prices** using a trained **Machine Learning model**.

Enter property details in the sidebar and click **Predict Price**.
""")

st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.header("🏡 Property Details")

MedianIncome = st.sidebar.number_input(
    "Median Income ($ per year)",
    min_value=10000,
    max_value=200000,
    value=60000,
    step=1000
)

HouseAge = st.sidebar.slider(
    "House Age (Years)",
    1,60,20
)

Rooms = st.sidebar.slider(
    "Total Rooms",
    1,12,5
)

Bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,6,2
)

Population = st.sidebar.number_input(
    "Population of Area",
    100,20000,3000,100
)

Occupancy = st.sidebar.slider(
    "People per House",
    1,8,3
)

Latitude = st.sidebar.number_input(
    "Latitude",
    32.0,42.0,36.7
)

Longitude = st.sidebar.number_input(
    "Longitude",
    -125.0,-114.0,-119.4
)

# ---------------- Contact Section ----------------
st.sidebar.markdown("---")
st.sidebar.header("📞 Contact")

st.sidebar.write("**Name:** Amit Gaikwad")
st.sidebar.write("**Email:** amitgaikwad2003@gmail.com")
st.sidebar.write("**Phone:** 9552475715")

# ---------------- Convert Input ----------------
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

# ---------------- Prediction Section ----------------
st.subheader("📊 Prediction")

if st.button("Predict Price"):

    prediction = model.predict(input_data)[0]
    price = prediction * 100000

    monthly_rent = price * 0.005
    yearly_rent = monthly_rent * 12

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🏠 Estimated House Price", f"${price:,.0f}")

    with col2:
        st.metric("💰 Estimated Monthly Rent", f"${monthly_rent:,.0f}")

    # ---------------- Chart ----------------
    chart_data = pd.DataFrame({
        "Category": ["House Price", "Yearly Rent"],
        "Amount": [price, yearly_rent]
    })

    st.subheader("📈 Price vs Rent Chart")
    st.bar_chart(chart_data.set_index("Category"))

# ---------------- Input Summary ----------------
st.markdown("---")
st.subheader("📋 Input Summary")

summary = {
    "Median Income ($)": MedianIncome,
    "House Age": HouseAge,
    "Rooms": Rooms,
    "Bedrooms": Bedrooms,
    "Population": Population,
    "People per House": Occupancy,
    "Latitude": Latitude,
    "Longitude": Longitude
}

st.write(summary)

# ---------------- Footer ----------------
st.markdown("---")
st.write("🚀 Built with Streamlit and Machine Learning")
