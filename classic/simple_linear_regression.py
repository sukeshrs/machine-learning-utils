import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
x = np.random.rand(100, 1) * 10  # Random x values between 0 and 10
y = 3 * x + 7 + np.random.randn(100, 1)  # Linear relation with noise

# Visualize the data
plt.scatter(x, y, color='blue', alpha=0.6)
plt.title("Scatter Plot of Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Linear Regression Implementation


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0  # Initialize weight
        self.b = 0  # Initialize bias

    def fit(self, x, y):
        n = len(x)
        for _ in range(self.epochs):
            # Predictions
            y_pred = self.w * x + self.b

            # Compute gradients
            dw = (-2 / n) * np.sum(x * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, x):
        return self.w * x + self.b


# Train the model
model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(x, y)

# Make predictions
y_pred = model.predict(x)

# Plot the results
plt.scatter(x, y, color='blue', alpha=0.6, label="Data")
plt.plot(x, y_pred, color='red', label="Best Fit Line")
plt.title("Linear Regression Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Display the learned parameters
print(f"Learned weight (w): {model.w}")
print(f"Learned bias (b): {model.b}")


