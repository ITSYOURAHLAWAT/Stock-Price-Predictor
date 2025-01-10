import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Streamlit app title
st.title("Stock Price Predictor")

# Input for stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL for Apple):", "AAPL")

# Fetch stock data
if st.button("Predict Stock Prices"):
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")

    # Prepare data for prediction
    data['Date'] = data.index
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)
    X = np.array(data['Date']).reshape(-1, 1)
    y = np.array(data['Close'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Display results
    st.line_chart(data['Close'])
    st.write("Actual vs Predicted Prices:")
    result_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    st.write(result_df)
start_date = st.date_input("Start Date", value=pd.Timestamp("2018-01-01"))
end_date = st.date_input("End Date", value=pd.Timestamp.today())

stock = yf.Ticker(ticker)
data = stock.history(start=start_date, end=end_date)
st.write("Stock Data Statistics:")
st.write(data.describe())
future_dates = np.array([data['Date'].iloc[-1] + i for i in range(1, 31)]).reshape(-1, 1)
future_predictions = model.predict(future_dates)
st.write("Future Prices (Next 30 Days):")
st.line_chart(future_predictions)
