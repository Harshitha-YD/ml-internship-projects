# Task 1: Data Preprocessing for Machine Learning

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script location
file_path = os.path.join(script_dir, "iris.csv")         # Build full path to iris.csv
print("Loading file from:", file_path)

df = pd.read_csv(file_path)
print("\n📊 First 5 rows of the dataset:")
print(df.head())

# Step 2: Check and handle missing data
print("\n🔍 Checking for missing values:")
print(df.isnull().sum())

# If there were missing values, you could fill them:
# df.fillna(df.mean(numeric_only=True), inplace=True)

# Step 3: Encode categorical variables (target column)
print("\n🔡 Encoding 'species' column...")
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
print("✅ Encoded classes:", le.classes_)  # Should be ['setosa' 'versicolor' 'virginica']

# Step 4: Standardize features
print("\n📏 Standardizing numeric features...")
X = df.drop('species', axis=1)   # features
y = df['species']                # target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the dataset
print("\n✂️ Splitting dataset into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\n✅ Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


