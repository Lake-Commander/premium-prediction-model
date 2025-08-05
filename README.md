# ⚠️ Disclaimer
This insurance premium prediction project is intended strictly for **educational purposes** only.
The model, predictions, and results showcased in this repository should not be used for any real-world insurance decision-making.
All data used is either synthetic or anonymized and does not represent actual insurance records.

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

```
premium-prediction-model/
├── models/
├── output_graphs/
│   ├── bivariate/
│   ├── eda/
│   └── feature_importance/
├── scripts/
├── utils/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
└── ...
```

**Descriptions:**
- `models/`: Saved machine learning models  
- `output_graphs/`: Visualizations (EDA, Feature Importance, etc.)  
- `scripts/`: Python scripts for preprocessing, training, etc.  
- `utils/`: Helper modules and functions  


---

## 🧠 Train Your Own Model

To train your own insurance premium prediction model using this project:

### Open the training notebook

You’ll find a Jupyter Notebook a.ipynb in the main project root.

### Run the notebook cells
The notebook will walk you through:

1. Loading and preprocessing the dataset

2. Feature engineering and transformations

3. Model training

4. Evaluation and saving of trained models to the models/ folder

### Use the trained model
Once training is complete, the saved models can be used by the app.py script to make predictions via the Streamlit interface.

> ℹ️ Ensure your dataset follows the expected schema. For reference, check the sample dataset (Insurance Premium Prediction Dataset.csv) used in this training.

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/lake-commander/premium-prediction-model.git
cd premium-prediction-model
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scriptsctivate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

Then go to [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Live Demo

You can also try out the app live on **Streamlit Cloud**:

👉 [Click here to open the app](https://premium-prediction-model.streamlit.app/)


## 🧪 Example Use Case

Given a user's input like:

- **Age**: 35  
- **Health Score**: 7.5  
- **Annual Income**: ₦3,500,000  
- **Vehicle Age**: 3 years  
- **Number of Dependents**: 2  
- **Claims History**: 1 claim  
- **Lifestyle**: Smoker, Urban Resident  

➡️ The model will predict an appropriate premium amount.

---

## 🙏 Acknowledgments

This project was built under the guidance and mentorship of the **3MTT (Three Million Technical Talent)** program by the **National Information Technology Development Agency (NITDA), Nigeria**.

We sincerely appreciate **NITDA** and the **Federal Ministry of Communications, Innovation and Digital Economy** for the opportunity to learn, grow, and contribute to Nigeria’s digital transformation journey.

> Thank you for empowering Nigerian youths with the skills to build real-world solutions.

---

## 🧠 Skills Demonstrated

- Data Cleaning and Transformation  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Regression Modeling  
- Model Evaluation and Selection  
- Streamlit App Development  
- Git & GitHub Collaboration  

---

## 🔗 License

This project is for **educational and demonstration purposes only**.

---

## 💬 Contact

Feel free to reach out if you have any feedback or questions!
