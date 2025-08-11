import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heart.csv")

plt.style.use('seaborn-v0_8-darkgrid')

# 1. Histogram for Age Distribution
plt.figure(figsize=(6,4))
plt.hist(df['age'], bins=15, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Bar chart for Gender Count
plt.figure(figsize=(5,4))
gender_counts = df['sex'].value_counts()
plt.bar(['Male', 'Female'], gender_counts, color=['steelblue', 'lightcoral'])
plt.title('Gender Distribution')
plt.ylabel('Count')
plt.show()

# 3. Scatter plot: Age vs. Maximum Heart Rate
plt.figure(figsize=(6,4))
plt.scatter(df['age'], df['thalach'], c=df['target'], cmap='coolwarm', alpha=0.7)
plt.colorbar(label='Target (0=No Disease, 1=Disease)')
plt.title('Age vs. Maximum Heart Rate')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate (thalach)')
plt.show()

# 4. Boxplot for Cholesterol by Target
plt.figure(figsize=(6,4))
df.boxplot(column='chol', by='target', grid=False)
plt.title('Cholesterol Levels by Heart Disease Status')
plt.suptitle('')
plt.xlabel('Heart Disease (0=No, 1=Yes)')
plt.ylabel('Cholesterol (mg/dl)')
plt.show()


