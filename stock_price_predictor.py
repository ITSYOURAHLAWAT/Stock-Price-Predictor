import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch stock data
stock = yf.Ticker("AAPL")
data = stock.history(period="5y")

# Print and plot data
print(data.head())
data['Close'].plot(title="Stock Closing Prices")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data for prediction
data['Date'] = data.index
data['Date'] = data['Date'].map(pd.Timestamp.toordinal)  # Convert dates to numbers
X = np.array(data['Date']).reshape(-1, 1)  # Features (Dates)
y = np.array(data['Close'])  # Target (Prices)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict stock prices
predictions = model.predict(X_test)

# Plot actual vs predicted prices
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, predictions, color='red', label='Predicted Prices')
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.show()
