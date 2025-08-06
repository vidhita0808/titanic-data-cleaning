# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
df = pd.read_csv("titanic-dataset.csv", encoding='latin1')  # Make sure the CSV is in the same folder

# Display the first few rows
print("\n🔹 First 5 Rows of Data:")
print(df.head())

# Summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe(include='all'))

# Missing values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Data types
print("\n🔹 Data Types:")
print(df.dtypes)

# Value counts for categorical variables
print("\n🔹 Value Counts for 'Sex':")
print(df['Sex'].value_counts())

print("\n🔹 Value Counts for 'Embarked':")
print(df['Embarked'].value_counts())

# ------------------------
# 📊 Visualizations
# ------------------------

# Histogram - Age and Fare
df[['Age', 'Fare']].hist(bins=30, figsize=(10, 5))
plt.suptitle("Histograms of Age and Fare", fontsize=14)
plt.tight_layout()
plt.show()

# Boxplots - Age and Fare
for col in ['Age', 'Fare']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Count plot for Survived
sns.countplot(x='Survived', data=df)
plt.title("Count of Survival (0 = No, 1 = Yes)")
plt.show()

# Count plot by gender and survival
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Gender")
plt.show()

# Heatmap - Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pie chart - Embarked
embarked_counts = df['Embarked'].value_counts()
plt.pie(embarked_counts, labels=embarked_counts.index, autopct='%1.1f%%')
plt.title("Passenger Embarkation Distribution")
plt.show()

# Interactive Plotly: Age vs Fare
fig = px.scatter(df, x="Age", y="Fare", color="Survived", title="Age vs Fare (colored by Survival)")
fig.show()
