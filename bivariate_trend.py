import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load dataset ===
df = pd.read_csv("transformed.csv")

# === Output directory ===
output_dir = "output_graphs/bivariate"
os.makedirs(output_dir, exist_ok=True)

# === Set plot style ===
sns.set(style="whitegrid")
target_col = "Premium Amount"

# === Identify column types ===
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Remove target from numerical features
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

# --- 1. Correlation heatmap ---
corr = df[numerical_cols + [target_col]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

# --- 2. Boxplots: Categorical vs Target ---
for col in categorical_cols:
    if df[col].nunique() <= 10:  # Avoid high-cardinality plots
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=col, y=target_col)
        plt.title(f"{col} vs {target_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/boxplot_{col}.png")
        plt.close()

# --- 3. Scatterplots: Numerical vs Target ---
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=col, y=target_col, alpha=0.5)
    plt.title(f"{col} vs {target_col}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatterplot_{col}.png")
    plt.close()

print("✅ Bivariate EDA plots saved to:", output_dir)
