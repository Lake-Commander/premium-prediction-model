import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load cleaned dataset
df = pd.read_csv("transformed.csv")

# Create output directory for graphs
output_dir = "output_graphs"
os.makedirs(output_dir, exist_ok=True)

# ---------------------- Correlation Matrix ----------------------
numerical_cols = [
    "Age", "Annual Income", "Number of Dependents", "Health Score",
    "Previous Claims", "Vehicle Age", "Credit Score",
    "Insurance Duration", "Premium Amount"
]

plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("🔗 Correlation Matrix (Numerical Features)")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_matrix.png")
plt.close()

# ---------------------- Boxplots: Premium vs Categorical ----------------------
categorical_cols = [
    "Gender", "Marital Status", "Education Level", "Occupation",
    "Location", "Policy Type", "Customer Feedback", "Smoking Status",
    "Exercise Frequency", "Property Type"
]

for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=col, y="Premium Amount", data=df, palette="Set3")
    plt.title(f"📦 Premium Amount by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"boxplot_premium_vs_{col}".lower().replace(" ", "_")
    plt.savefig(f"{output_dir}/{filename}.png")
    plt.close()

print(f"✅ Correlation and boxplot graphs saved to ./{output_dir}")
