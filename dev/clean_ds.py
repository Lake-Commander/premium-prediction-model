import pandas as pd

# Load dataset
df = pd.read_csv("Insurance Premium Prediction Dataset.csv")

# Fill missing numeric values with median
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Annual Income"].fillna(df["Annual Income"].median(), inplace=True)
df["Number of Dependents"].fillna(df["Number of Dependents"].median(), inplace=True)
df["Credit Score"].fillna(df["Credit Score"].median(), inplace=True)
df["Health Score"].fillna(df["Health Score"].median(), inplace=True)
df["Premium Amount"].fillna(df["Premium Amount"].median(), inplace=True)

# Fill missing categorical values
df["Occupation"].fillna("Unknown", inplace=True)
df["Customer Feedback"].fillna(df["Customer Feedback"].mode()[0], inplace=True)
df["Marital Status"].fillna(df["Marital Status"].mode()[0], inplace=True)

# Fill Previous Claims (assumed as no claims)
df["Previous Claims"].fillna(0.0, inplace=True)

# Optional: Save cleaned dataset
df.to_csv("cleaned_insurance_data.csv", index=False)

# Preview result
print("✅ Dataset cleaned successfully.")
print("Shape:", df.shape)
print("Remaining missing values:")
print(df.isnull().sum())
