import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("transformed.csv")

# ========== 1. NUMERICAL vs NUMERICAL ==========
# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[["Annual Income", "Health Score", "Credit Score", "Premium Amount"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/bivariate_correlation_heatmap.png")
plt.close()

# Scatterplot: Income vs Premium
sns.scatterplot(data=df, x="Annual Income", y="Premium Amount", alpha=0.4)
plt.title("Annual Income vs Premium Amount")
plt.tight_layout()
plt.savefig("plots/scatter_income_premium.png")
plt.close()

# ========== 2. CATEGORICAL vs NUMERICAL ==========
cat_num_pairs = [
    ("Gender", "Premium Amount"),
    ("Education Level", "Premium Amount"),
    ("Occupation", "Premium Amount"),
    ("Smoking Status", "Premium Amount"),
]

for cat, num in cat_num_pairs:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=cat, y=num, palette="Set3")
    plt.xticks(rotation=45)
    plt.title(f"{cat} vs {num}")
    plt.tight_layout()
    plt.savefig(f"plots/box_{cat.lower().replace(' ', '_')}_vs_{num.lower().replace(' ', '_')}.png")
    plt.close()

# ========== 3. CATEGORICAL vs CATEGORICAL ==========
cat_cat_pairs = [
    ("Gender", "Smoking Status"),
    ("Marital Status", "Exercise Frequency"),
    ("Occupation", "Property Type"),
]

for cat1, cat2 in cat_cat_pairs:
    cross_tab = pd.crosstab(df[cat1], df[cat2])
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"{cat1} vs {cat2}")
    plt.tight_layout()
    plt.savefig(f"plots/heatmap_{cat1.lower().replace(' ', '_')}_vs_{cat2.lower().replace(' ', '_')}.png")
    plt.close()

print("✅ Bivariate analysis completed and plots saved to 'plots/' folder.")
