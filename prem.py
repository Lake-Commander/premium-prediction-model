import pandas as pd

# Load the dataset (make sure the file name is correct)
df = pd.read_csv("Typed_Insurance_Dataset.csv")

# Basic inspection
print("Shape of dataset:", df.shape)
print("\nColumn names and data types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nSample data:\n", df.head())
