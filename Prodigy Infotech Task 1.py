# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Converting the data to a DataFrame
df = pd.read_csv('D:/Engineering/Prodigy Infotech/Prodigy Infotech Task 1/ProgidyTask1/Housing.csv')

# Splitting the dataset into training and testing sets
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Linear Regression model
model = LinearRegression()

# Training the model on the training data
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# new data
new_data = pd.DataFrame({'SquareFootage': [1800], 'Bedrooms': [3], 'Bathrooms': [2]})
predicted_price = model.predict(new_data)
print(f"Predicted price for new data: ${predicted_price[0]:,.2f}")
