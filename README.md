# 🏠 California Housing Price Prediction using XGBoost

A **Machine Learning web application** built with **Streamlit** that predicts California house prices based on housing features such as median income, house age, population, and location.

The model is trained using the **California Housing Dataset** and the **XGBoost Regressor algorithm**.

---

## 🌐 Live Demo

Try the application here:

🔗 **Streamlit App:**
https://xgboost-housing-price-predictor-hdt5nszq76itqmhchuzm5u.streamlit.app/

This live application allows users to enter housing details and instantly predict house prices using a trained machine learning model.

---

## 📌 Project Overview

This project demonstrates how **Machine Learning can be applied to real estate price prediction**.

Users can enter housing details through an **interactive web interface**, and the trained model predicts the **estimated house price in real time**.

The application uses **Streamlit** to provide a simple and user-friendly interface for testing predictions.

---

## 📊 Dataset

The model is trained using the **California Housing Dataset** provided by **scikit-learn**.

### Features Used

| Feature    | Description                  |
| ---------- | ---------------------------- |
| MedInc     | Median income in block group |
| HouseAge   | Median house age             |
| AveRooms   | Average number of rooms      |
| AveBedrms  | Average number of bedrooms   |
| Population | Block population             |
| AveOccup   | Average house occupancy      |
| Latitude   | House location latitude      |
| Longitude  | House location longitude     |

**Target Variable:**
Median House Value (Predicted House Price)

---

## ⚙️ Technologies Used

* Python
* Streamlit
* XGBoost
* Scikit-learn
* NumPy
* Pandas
* Pickle

---

## 📂 Project Structure

california-housing-price-predictor/

│
├── app.py (Streamlit application)
├── model_xgb.pkl (Trained XGBoost model)
├── requirements.txt (Project dependencies)
└── README.md (Project documentation)

---

## 🚀 How to Run the Project

### 1. Clone the Repository

git clone https://github.com/Vishal132003/california-housing-price-predictor.git

### 2. Navigate to the Project Folder

cd california-housing-price-predictor

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the Streamlit Application

streamlit run app.py

---

## 🖥️ Application Features

• Interactive sidebar for entering housing details
• Real-time house price prediction
• Clean and simple Streamlit interface
• Input summary display
• Contact information in sidebar

---

## 📈 Model Information

**Model Used:** XGBoost Regressor

Why XGBoost?

• High predictive performance
• Handles non-linear relationships effectively
• Efficient and scalable
• Widely used in real-world machine learning projects

The trained model is saved using **Pickle** and loaded inside the Streamlit application for making predictions.

---

## 👨‍💻 Author

**Vishal Jadhav**

Email: [vishaljadhav132003@gmail.com](mailto:vishaljadhav132003@gmail.com)
Phone: 9529935831
