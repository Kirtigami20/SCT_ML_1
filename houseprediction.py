# Linear Regression for House Price Prediction
# Using GrLivArea, BedroomAbvGr, and FullBath

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("train.csv")

# Select relevant features
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
df = df[features + ["SalePrice"]].dropna()

# Split data
X = df[features]
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
print("Intercept:", model.intercept_)
for fname, coef in zip(features, model.coef_):
    print(f"{fname}: {coef:.2f}")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualization - Predicted vs Actual
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Example prediction for new house
new_house = [[2000, 3, 2]]  # 2000 sqft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(new_house)
print("\nPredicted Price for new house:", predicted_price[0])
