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
Folder PATH listing
Volume serial number is 9A68-80D0
C:.
│   .gitattributes
│   a.ipynb
│   app.py
│   cleaned_insurance_data.csv
│   Insurance Premium Prediction Dataset.csv
│   LICENSE
│   processed_rf_selected_data.csv
│   random_forest_model.pkl
│   README.md
│   requirements.txt
│   transformed.csv
│   tree.txt
│   Typed_Insurance_Dataset.csv
│   X_test.npy
│   X_train.npy
│   y_test.npy
│   y_train.npy
│   
├───bivariate_analysis
│       bivariate_correlation_heatmap.png
│       box_education_level_vs_premium_amount.png
│       box_gender_vs_premium_amount.png
│       box_occupation_vs_premium_amount.png
│       box_smoking_status_vs_premium_amount.png
│       heatmap_gender_vs_smoking_status.png
│       heatmap_marital_status_vs_exercise_frequency.png
│       heatmap_occupation_vs_property_type.png
│       scatter_income_premium.png
│       
├───dev
│       app.py
│       bivariate_analysis.py
│       bivariate_trend.py
│       clean_ds.py
│       data_types.py
│       feature.py
│       model_training.py
│       model_tuning.py
│       multivariate_analysis.py
│       multivariate_trend.py
│       prem.py
│       premium_trend_analysis.py
│       skewed.py
│       univariate_analysis.py
│       
├───models
│   │   feature_order.pkl
│   │   random_forest_model.pkl
│   │   scaler.pkl
│   │   
│   └───check
│           random_forest_model.pkl
│           
├───multivariate
│       correlation_heatmap.png
│       groupedbar_education_level_marital_status_by_gender.png
│       groupedbar_location_property_type_by_policy_type.png
│       groupedbar_smoking_status_exercise_frequency_by_policy_type.png
│       heatmap_customer_feedback_smoking_status.png
│       heatmap_education_level_occupation.png
│       heatmap_gender_policy_type.png
│       heatmap_location_property_type.png
│       pairplot_numeric.png
│       premium_by_education_level.png
│       premium_by_exercise_frequency.png
│       premium_by_gender.png
│       premium_by_marital_status.png
│       premium_by_occupation.png
│       premium_by_policy_type.png
│       premium_by_property_type.png
│       premium_by_smoking_status.png
│       
├───output_graphs
│   ├───bivariate
│   │       boxplot_Customer Feedback.png
│   │       boxplot_Education Level.png
│   │       boxplot_Exercise Frequency.png
│   │       boxplot_Gender.png
│   │       boxplot_Location.png
│   │       boxplot_Marital Status.png
│   │       boxplot_Occupation.png
│   │       boxplot_Policy Type.png
│   │       boxplot_Property Type.png
│   │       boxplot_Smoking Status.png
│   │       correlation_heatmap.png
│   │       scatterplot_Age.png
│   │       scatterplot_Annual Income.png
│   │       scatterplot_Annual Income_log.png
│   │       scatterplot_Credit Score.png
│   │       scatterplot_Health Score.png
│   │       scatterplot_Insurance Duration.png
│   │       scatterplot_Number of Dependents.png
│   │       scatterplot_Premium Amount_log.png
│   │       scatterplot_Previous Claims.png
│   │       scatterplot_Previous Claims_log.png
│   │       scatterplot_Vehicle Age.png
│   │       
│   └───multivariate
│           Age_vs_Premium Amount_regplot.png
│           Annual Income_log_vs_Premium Amount_regplot.png
│           Annual Income_vs_Premium Amount_regplot.png
│           Credit Score_vs_Premium Amount_regplot.png
│           Customer Feedback_vs_Premium Amount_boxplot.png
│           Education Level_vs_Premium Amount_boxplot.png
│           Exercise Frequency_vs_Premium Amount_boxplot.png
│           Gender_vs_Premium Amount_boxplot.png
│           Health Score_vs_Premium Amount_regplot.png
│           Insurance Duration_vs_Premium Amount_regplot.png
│           Location_vs_Premium Amount_boxplot.png
│           Marital Status_vs_Premium Amount_boxplot.png
│           Number of Dependents_vs_Premium Amount_regplot.png
│           Occupation_vs_Premium Amount_boxplot.png
│           Policy Type_vs_Premium Amount_boxplot.png
│           Premium Amount_log_vs_Premium Amount_regplot.png
│           Previous Claims_log_vs_Premium Amount_regplot.png
│           Previous Claims_vs_Premium Amount_regplot.png
│           Property Type_vs_Premium Amount_boxplot.png
│           Smoking Status_vs_Premium Amount_boxplot.png
│           Vehicle Age_vs_Premium Amount_regplot.png
│           
├───output_graphs_unused
│   └───bivariate
│           correlation_heatmap.png
│           premium_amount_by_education_level.png
│           premium_amount_by_gender.png
│           premium_amount_by_location.png
│           premium_amount_by_marital_status.png
│           premium_amount_by_occupation.png
│           premium_amount_by_policy_type.png
│           premium_amount_vs_age.png
│           premium_amount_vs_annual_income.png
│           premium_amount_vs_annual_income_log.png
│           premium_amount_vs_credit_score.png
│           premium_amount_vs_health_score.png
│           premium_amount_vs_insurance_duration.png
│           premium_amount_vs_number_of_dependents.png
│           premium_amount_vs_premium_amount_log.png
│           premium_amount_vs_previous_claims.png
│           premium_amount_vs_previous_claims_log.png
│           premium_amount_vs_vehicle_age.png
│           
├───premium_trend_correlation_categorical
│       boxplot_premium_vs_customer_feedback.png
│       boxplot_premium_vs_education_level.png
│       boxplot_premium_vs_exercise_frequency.png
│       boxplot_premium_vs_gender.png
│       boxplot_premium_vs_location.png
│       boxplot_premium_vs_marital_status.png
│       boxplot_premium_vs_occupation.png
│       boxplot_premium_vs_policy_type.png
│       boxplot_premium_vs_property_type.png
│       boxplot_premium_vs_smoking_status.png
│       correlation_matrix.png
│       
├───transformations
│       after_log_transform.png
│       bfr_log_transform.png
│       
└───univariate
        count_customer_feedback.png
        count_education_level.png
        count_exercise_frequency.png
        count_gender.png
        count_location.png
        count_marital_status.png
        count_occupation.png
        count_policy_type.png
        count_property_type.png
        count_smoking_status.png
        hist_age.png
        hist_annual_income.png
        hist_credit_score.png
        hist_health_score.png
        hist_insurance_duration.png
        hist_number_of_dependents.png
        hist_premium_amount.png
        hist_previous_claims.png
        hist_vehicle_age.png

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/premium-prediction-model.git
cd premium-prediction-model

### 2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Streamlit App
streamlit run streamlit_app.py

'''bash

go to http://localhost:8501

## 🧪 Example Use Case
Given a user's input like:

Age: 35

Health Score: 7.5

Annual Income: ₦3,500,000

Vehicle Age: 3 years

Number of Dependents: 2

Claims History: 1 claim

Lifestyle: Smoker, Urban Resident

The model will predict an appropriate premium amount, e.g., ₦235,000.

## 🙏 Acknowledgments
This project was built under the guidance and mentorship of the 3MTT (Three Million Technical Talent) program by the National Information Technology Development Agency (NITDA), Nigeria.

We sincerely appreciate NITDA and the Federal Ministry of Communications, Innovation and Digital Economy for the opportunity to learn, grow, and contribute to Nigeria’s digital transformation journey.

Thank you for empowering Nigerian youths with the skills to build real-world solutions.

## 🧠 Skills Demonstrated
Data Cleaning and Transformation

Exploratory Data Analysis (EDA)

Feature Engineering

Regression Modeling

Model Evaluation and Selection

Streamlit App Development

Git & GitHub Collaboration

## 🔗 License
This project is for educational and demonstration purposes only.

## 💬 Contact
Feel free to reach out if you have any feedback or questions!
