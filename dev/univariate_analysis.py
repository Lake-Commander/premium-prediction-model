import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("transformed.csv")  # Replace with your cleaned file path

# Set seaborn style
sns.set(style="whitegrid")

# Create directory for saving plots
output_dir = "plots/univariate"
os.makedirs(output_dir, exist_ok=True)

# ----------- Numerical Features -----------
numerical_features = ["Age", "Annual Income", "Number of Dependents", "Health Score",
                      "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration",
                      "Premium Amount"]

# Summary statistics
print("🔢 Summary Statistics:")
print(df[numerical_features].describe())

# Histograms for numerical features
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    safe_name = col.lower().replace(" ", "_")
    plt.savefig(f"{output_dir}/hist_{safe_name}.png")
    plt.close()

# ----------- Categorical Features -----------
categorical_features = ["Gender", "Marital Status", "Education Level", "Occupation",
                        "Location", "Policy Type", "Customer Feedback", "Smoking Status",
                        "Exercise Frequency", "Property Type"]

# Count plots for categorical features
for col in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, palette="Set2", order=df[col].value_counts().index)
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    safe_name = col.lower().replace(" ", "_")
    plt.savefig(f"{output_dir}/count_{safe_name}.png")
    plt.close()

print(f"✅ Univariate plots saved in: {output_dir}/")
