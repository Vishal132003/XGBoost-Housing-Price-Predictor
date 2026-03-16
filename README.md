# 🏠 California Housing Price Prediction using XGBoost

A **Machine Learning web application** built with **Streamlit** that predicts California house prices based on housing features such as median income, house age, population, and location.
The model is trained using the California Housing dataset and an XGBoost regression algorithm.

---

## 📌 Project Overview

This project demonstrates how Machine Learning can be used to estimate real estate prices.
Users can enter housing details through a simple web interface, and the trained model will predict the estimated house price.

The application is designed with a **clean interactive interface using Streamlit**, making it easy for users to test predictions in real time.

---

## 📊 Dataset

The model is trained using the California Housing dataset provided by **scikit-learn**.

**Features used in the model:**

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

**Target Variable**

* Median house value (predicted price)

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

```
california-housing-price-predictor/
│
├── app.py                # Streamlit application
├── model_xgb.pkl         # Trained XGBoost model
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/california-housing-price-predictor.git
```

### 2️⃣ Navigate to the project folder

```
cd california-housing-price-predictor
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit application

```
streamlit run app.py
```

---

## 🖥️ Application Features

* Interactive sidebar inputs
* Real-time house price prediction
* Clean and simple Streamlit interface
* Input summary display
* Contact details in sidebar

---

## 📈 Model Information

Model Used: **XGBoost Regressor**

Why XGBoost?

* High performance
* Handles non-linear relationships well
* Robust and widely used in real-world ML applications

The trained model is saved using **Pickle** and loaded inside the Streamlit application for prediction.

---

## 👨‍💻 Author

**Vishal Jadhav**

📧 Email: [vaishnavijadhav01234@gmail.com](mailto:vaishnavijadhav01234@gmail.com)
📱 Phone: 8788965221
🔗 LinkedIn: https://www.linkedin.com/in/vaishnavi-jadhav-465774327

---

## 📜 License

This project is developed for **educational and portfolio purposes**.
