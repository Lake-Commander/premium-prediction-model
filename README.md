# 🛡️ Insurance Premium Prediction Model

This project presents an intelligent system that predicts **insurance premium amounts** based on customer profiles using a machine learning model. It aims to support insurance companies in **pricing policies more accurately**, minimizing risk, and improving customer segmentation.

The entire project pipeline includes data preprocessing, exploratory analysis, feature engineering, model building, evaluation, and deployment using **Streamlit**.

---

## 🎯 Project Objective

To develop and deploy a machine learning model that:
- Predicts the **premium amount** a customer is likely to pay.
- Uses customer features such as income, age, vehicle age, claims history, health status, and more.
- Helps insurance companies make **data-driven pricing decisions**.
- Offers an **interactive Streamlit web app** for easy use by non-technical users.

---

## 🧠 Machine Learning Approach

- **Target Variable:** Premium Amount (log-transformed)
- **Algorithms Used:**
  - Linear Regression
  - Ridge & Lasso Regression
  - Random Forest Regressor (final deployed model)
- **Evaluation Metrics:**
  - Mean Squared Error (MSE)
  - R² Score
- **Feature Scaling:** MinMaxScaler
- **Encoding:** One-hot encoding for categorical features

---

## 📦 Project Structure

# 🛡️ Insurance Premium Prediction Model

This project presents an intelligent system that predicts **insurance premium amounts** based on customer profiles using a machine learning model. It aims to support insurance companies in **pricing policies more accurately**, minimizing risk, and improving customer segmentation.

The entire project pipeline includes data preprocessing, exploratory analysis, feature engineering, model building, evaluation, and deployment using **Streamlit**.

---

## 🎯 Project Objective

To develop and deploy a machine learning model that:
- Predicts the **premium amount** a customer is likely to pay.
- Uses customer features such as income, age, vehicle age, claims history, health status, and more.
- Helps insurance companies make **data-driven pricing decisions**.
- Offers an **interactive Streamlit web app** for easy use by non-technical users.

---

## 🧠 Machine Learning Approach

- **Target Variable:** Premium Amount (log-transformed)
- **Algorithms Used:**
  - Linear Regression
  - Ridge & Lasso Regression
  - Random Forest Regressor (final deployed model)
- **Evaluation Metrics:**
  - Mean Squared Error (MSE)
  - R² Score
- **Feature Scaling:** MinMaxScaler
- **Encoding:** One-hot encoding for categorical features

---

## 📦 Project Structure

premium-prediction-model/
│
├── streamlit_app.py # Streamlit app for prediction
├── model.py # Training and evaluation of models
├── scaler.pkl # Trained MinMaxScaler
├── final_model.pkl # Trained Random Forest Regressor
├── requirements.txt # Python dependencies
├── transformed_dataset.csv # Cleaned and preprocessed dataset
├── README.md # Project documentation
└── 📁output_graphs/ # EDA visualizations (optional)


---

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/premium-prediction-model.git
cd premium-prediction-model

### 2. Create a Virtual Environment


### 3. Install Dependencies
