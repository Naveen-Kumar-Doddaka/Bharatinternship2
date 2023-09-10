import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a synthetic dataset for demonstration
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Random feature (e.g., square footage)
y = 2 * X + 1 + np.random.randn(100, 1)  # House price with some noise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict the house price for a new data point
new_data_point = np.array([[5.0]])  # You can change this value
predicted_price = model.predict(new_data_point)
print("Predicted house price for new data point:", predicted_price[0][0])
