import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("transformed.csv")

# Set seaborn style
sns.set(style="white")

# Create output directory
output_dir = "plots/multivariate"
os.makedirs(output_dir, exist_ok=True)

# Numerical columns for correlation
numerical_features = ["Age", "Annual Income", "Number of Dependents", "Health Score",
                      "Previous Claims", "Vehicle Age", "Credit Score", "Insurance Duration",
                      "Premium Amount"]

# ----------- 1. Correlation Heatmap -----------
plt.figure(figsize=(10, 8))
corr_matrix = df[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

# ----------- 2. Pairplot (Sampled for speed) -----------
sample_df = df[numerical_features].sample(2000, random_state=42)  # Reduce for speed/memory
sns.pairplot(sample_df)
plt.savefig(f"{output_dir}/pairplot_numeric.png")
plt.close()

# ----------- 3. Box plots vs. Premium Amount for selected categorical -----------
categorical_features = ["Gender", "Marital Status", "Education Level", "Occupation",
                        "Policy Type", "Smoking Status", "Exercise Frequency", "Property Type"]

for cat in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=cat, y="Premium Amount", palette="pastel")
    plt.xticks(rotation=45)
    plt.title(f"Premium Amount by {cat}")
    plt.tight_layout()
    safe_name = cat.lower().replace(" ", "_")
    plt.savefig(f"{output_dir}/premium_by_{safe_name}.png")
    plt.close()

    # ----------- 4. Grouped Bar Plots (Categorical vs Categorical with Hue) -----------

# Grouped barplot examples
grouped_combinations = [
    ("Smoking Status", "Exercise Frequency", "Policy Type"),
    ("Education Level", "Marital Status", "Gender"),
    ("Location", "Property Type", "Policy Type"),
]

for x_col, y_col, hue_col in grouped_combinations:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=x_col, hue=hue_col, order=df[x_col].value_counts().index, palette="Set2")
    plt.title(f"{x_col} vs {y_col} grouped by {hue_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    safe_name = f"{x_col}_{y_col}_by_{hue_col}".lower().replace(" ", "_")
    plt.savefig(f"{output_dir}/groupedbar_{safe_name}.png")
    plt.close()

    # ----------- 5. Heatmaps for Categorical Co-occurrence -----------

categorical_pairs = [
    ("Gender", "Policy Type"),
    ("Education Level", "Occupation"),
    ("Location", "Property Type"),
    ("Customer Feedback", "Smoking Status")
]

for row_col, col_col in categorical_pairs:
    cross_tab = pd.crosstab(df[row_col], df[col_col])
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
    plt.title(f"Co-occurrence Heatmap: {row_col} vs {col_col}")
    plt.xlabel(col_col)
    plt.ylabel(row_col)
    plt.tight_layout()
    
    safe_name = f"heatmap_{row_col}_{col_col}".lower().replace(" ", "_")
    plt.savefig(f"{output_dir}/{safe_name}.png")
    plt.close()



print(f"✅ Multivariate plots saved to: {output_dir}/")
