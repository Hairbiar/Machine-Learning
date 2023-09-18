import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)


# Define a function to perform linear regression
def linear_regression(X, y, learning_rate=0.01, num_iterations=1000):
    n = len(X)
    # Initialize the slope (m) and intercept (b)
    m = 0
    b = 0

    # Define the cost function (Mean Squared Error)
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Gradient descent to minimize the cost function
    for iteration in range(num_iterations):
        y_pred = m * X + b
        cost = mse(y, y_pred)

        # Calculate the gradients
        dm = (-2 / n) * np.sum(X * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)

        # Update the parameters
        m -= learning_rate * dm
        b -= learning_rate * db

        # Print the cost every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost = {cost}")

    return m, b


# Fit the model to the data
slope, intercept = linear_regression(X, y)

# Make predictions
X_new = np.array([[0], [2]])
y_pred = intercept + slope * X_new

# Plot the original data points
plt.scatter(X, y, label='Original Data')

# Plot the regression line
plt.plot(X_new, y_pred, 'r-', label='Regression Line')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print("Slope (m):", slope)
print("Intercept (b):", intercept)
