
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Check info and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df.drop(columns=['deck'], inplace=True)

# Convert categorical to numeric
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Visualize outliers
sns.boxplot(df['age'])
plt.title("Boxplot of Age")
plt.show()

# Remove outliers using IQR
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['age'] < (Q1 - 1.5 * IQR)) | (df['age'] > (Q3 + 1.5 * IQR)))]

# Normalize age and fare
scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# Final preview
print("\nPreprocessed Data:")
print(df.head())
