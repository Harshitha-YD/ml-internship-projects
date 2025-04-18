# house_prices_regression.py

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the CSV file with correct path
script_dir = os.path.dirname(os.path.abspath(__file__))  # current script location
file_path = os.path.join(script_dir, "house_prediction.csv")  # dataset in same folder
print("Loading file from:", file_path)

# Step 2: Define correct column names (Boston Housing Dataset)
columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"
]

# Step 3: Read the dataset - fix space-separated issue
df = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)


# Optional: Show first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Step 4: Choose feature (X) and target (y)
X = df[["RM"]]  # Average number of rooms
y = df["PRICE"]  # Median house price

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Step 6: Predictions
y_pred = model.predict(X)

# Step 7: Interpret the coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope (for RM): {model.coef_[0]:.2f}")

# Step 8: Evaluate model
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("\nModel Evaluation:")
print(f"R-squared: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
