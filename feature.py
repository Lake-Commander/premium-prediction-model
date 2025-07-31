import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("transformed.csv")

# Step 1: Feature Engineering
df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
df['Policy_Years_Since'] = datetime.now().year - df['Policy Start Date'].dt.year
df.drop(columns=['Policy Start Date'], inplace=True)

# Step 2: Separate target and features
target = 'Premium Amount'
X = df.drop(columns=[target])
y = df[target]

# Step 3: Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Step 4: Impute missing values
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Step 5: Encode categoricals
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_cat = encoder.fit_transform(X[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))

# Step 6: Scale numericals
scaler = StandardScaler()
scaled_num = scaler.fit_transform(X[numerical_cols])
scaled_num_df = pd.DataFrame(scaled_num, columns=numerical_cols)

# Step 7: Combine processed features
X_processed = pd.concat([scaled_num_df.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

# Step 8: Random Forest for Feature Selection
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_processed, y)

# Get feature importances
importances = rf.feature_importances_
feature_names = X_processed.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select top K features
k = 30  # Change this as needed
top_features = importance_df['Feature'].iloc[:k].tolist()
X_selected_df = X_processed[top_features]

# (Optional) Save selected data with target
X_selected_df[target] = y.reset_index(drop=True)
X_selected_df.to_csv("processed_rf_selected_data.csv", index=False)
