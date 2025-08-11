# ----- Import Libraries -----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----- Step 1: Load Data -----
df = pd.read_csv('housing.csv')
print("First 5 rows of dataset:")
print(df.head())

# ----- Step 2: SIMPLE Linear Regression -----
# We'll predict 'price' based on 'area' only
X_simple = df[['area']]    
y = df['price']           

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Model training
simple_model = LinearRegression()
simple_model.fit(X_train, y_train)

# Prediction
y_pred_simple = simple_model.predict(X_test)

# Evaluation
print("\n--- Simple Linear Regression ---")
print(f"MAE: {mean_absolute_error(y_test, y_pred_simple):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_simple):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_simple):.2f}")
print("Coefficient:", simple_model.coef_)
print("Intercept:", simple_model.intercept_)

# Plot
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_simple, color='red', linewidth=2, label='Predicted')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# ----- Step 3: MULTIPLE Linear Regression -----
# We'll predict 'price' using multiple features
# Encode categorical variables first
df_encoded = pd.get_dummies(df, drop_first=True)

# Features for multiple regression (excluding target 'price')
X_multiple = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

# Model training
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

# Prediction
y_pred_multi = multi_model.predict(X_test)

# Evaluation
print("\n--- Multiple Linear Regression ---")
print(f"MAE: {mean_absolute_error(y_test, y_pred_multi):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_multi):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_multi):.2f}")
print("Coefficients:", multi_model.coef_)
print("Intercept:", multi_model.intercept_)

