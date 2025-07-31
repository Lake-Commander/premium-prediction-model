import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIG
TARGET = 'Premium Amount'
INPUT_FILE = 'transformed.csv'
OUTPUT_DIR = 'output_graphs/multivariate'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD DATA
df = pd.read_csv(INPUT_FILE)
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# COLUMN TYPES
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop(TARGET, errors='ignore').tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

### 🔷 NUMERICAL FEATURES vs TARGET
corr = df[numerical_cols + [TARGET]].corr()[[TARGET]].drop(TARGET)
corr = corr.sort_values(by=TARGET, ascending=False)
print("📊 Numerical correlations:\n", corr)

for col in corr.index:
    plt.figure(figsize=(6, 4))
    sns.regplot(data=df, x=col, y=TARGET, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    plt.title(f'{col} vs {TARGET}')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{col}_vs_{TARGET}_regplot.png')
    plt.close()

### 🔶 CATEGORICAL FEATURES vs TARGET
for col in categorical_cols:
    if df[col].nunique() < 50:  # Skip too many-category columns
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=col, y=TARGET)
        plt.xticks(rotation=45)
        plt.title(f'{col} vs {TARGET}')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{col}_vs_{TARGET}_boxplot.png')
        plt.close()

print(f"✅ Multivariate plots saved to: {OUTPUT_DIR}")
