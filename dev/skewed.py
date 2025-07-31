import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("Typed_Insurance_Dataset.csv")

# List of numerical columns to check for skew
numeric_cols = [
    "Annual Income", "Credit Score", "Premium Amount",
    "Health Score", "Age", "Number of Dependents", "Previous Claims"
]

# Calculate skewness
skew_vals = df[numeric_cols].skew()

print("🔍 Skewness of numerical features:")
print(skew_vals)
print("\n📌 Highly skewed features (|skew| > 1):")

# Threshold for high skew
threshold = 1
high_skew_cols = skew_vals[abs(skew_vals) > threshold].index.tolist()
print(high_skew_cols)

# Apply log1p transformation to fix right skew
for col in high_skew_cols:
    # Skip columns with negative or NaN values
    if (df[col] < 0).any():
        print(f"⚠️ Skipping {col} due to negative values.")
        continue
    df[col + "_log"] = np.log1p(df[col])
    print(f"✅ Transformed {col} → {col}_log")

# Optional: visualize one transformed feature
if high_skew_cols:
    first = high_skew_cols[0]
    sns.histplot(df[first], kde=True)
    plt.title(f"{first} - Before Log Transform")
    plt.show()

    sns.histplot(df[first + "_log"], kde=True)
    plt.title(f"{first} - After Log Transform")
    plt.show()

# Save to new CSV for reference
df.to_csv("transformed.csv", index=False)
print("\n✅ Skewed features transformed and saved to 'cleaned_skew_fixed.csv'")
