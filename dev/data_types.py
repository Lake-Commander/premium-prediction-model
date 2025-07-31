import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("cleaned_insurance_data.csv")  # or your actual cleaned file

# Fix data types
df["Number of Dependents"] = df["Number of Dependents"].astype(int)
df["Previous Claims"] = df["Previous Claims"].astype(int)
df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], errors="coerce")

# Print results for confirmation
print("✅ Data types corrected successfully.")
print("\nUpdated Data Types:\n", df.dtypes)

# Save the corrected dataset (optional)
df.to_csv("Typed_Insurance_Dataset.csv", index=False)
