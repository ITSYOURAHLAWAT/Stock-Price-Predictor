import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, ticker, period="2y"):
        self.ticker = ticker
        self.period = period
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            print(f"✅ Successfully fetched {len(self.data)} days of data for {self.ticker}")
            return True
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators"""
        if self.data is None:
            print("❌ No data available. Please fetch data first.")
            return
        
        # Moving averages
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Price change percentage
        self.data['Price_Change'] = self.data['Close'].pct_change()
        
        print("✅ Technical indicators calculated")
    
    def create_features(self):
        """Create features for machine learning"""
        df = self.data.copy()
        
        # Price-based features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Price_Range'] = df['High'] - df['Low']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Close_MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_models(self):
        """Train prediction models"""
        if self.data is None:
            print("❌ No data available. Please fetch data first.")
            return
        
        # Create features
        featured_data = self.create_features()
        
        # Select feature columns
        feature_cols = [col for col in featured_data.columns 
                       if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']]
        
        X = featured_data[feature_cols]
        y = featured_data['Close']
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            # Store results
            results[name] = {
                'model': model,
                'predictions': predictions,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'test_dates': X_test.index,
                'actual': y_test
            }
            
            print(f"✅ {name} trained - R²: {r2:.4f}, RMSE: ${rmse:.2f}")
        
        self.models = results
        return results
    
    def plot_results(self):
        """Plot stock data and predictions"""
        if self.data is None or not self.models:
            print("❌ No data or models available.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.ticker} Stock Analysis', fontsize=16)
        
        # Plot 1: Stock price with moving averages
        axes[0, 0].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(self.data.index, self.data['MA_20'], label='MA 20', alpha=0.7)
        axes[0, 0].plot(self.data.index, self.data['MA_50'], label='MA 50', alpha=0.7)
        axes[0, 0].set_title('Stock Price & Moving Averages')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Volume
        axes[0, 1].bar(self.data.index, self.data['Volume'], alpha=0.6, color='orange')
        axes[0, 1].set_title('Trading Volume')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: RSI
        axes[1, 0].plot(self.data.index, self.data['RSI'], color='purple')
        axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[1, 0].set_title('RSI (Relative Strength Index)')
        axes[1, 0].set_ylabel('RSI')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Predictions comparison
        best_model = min(self.models.keys(), key=lambda x: self.models[x]['rmse'])
        model_data = self.models[best_model]
        
        axes[1, 1].plot(model_data['test_dates'], model_data['actual'], 
                       label='Actual', linewidth=2, color='blue')
        axes[1, 1].plot(model_data['test_dates'], model_data['predictions'], 
                       label=f'{best_model} Predicted', linewidth=2, color='red', linestyle='--')
        axes[1, 1].set_title(f'Actual vs Predicted ({best_model})')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=7):
        """Predict future stock prices"""
        if not self.models:
            print("❌ No trained models available.")
            return
        
        # Use the best performing model
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x]['rmse'])
        best_model = self.models[best_model_name]['model']
        
        print(f"🔮 Predicting next {days} days using {best_model_name}:")
        
        # Get last known features (simplified approach)
        featured_data = self.create_features()
        feature_cols = [col for col in featured_data.columns 
                       if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']]
        
        last_features = featured_data[feature_cols].iloc[-1:].values
        last_features_scaled = self.scaler.transform(last_features)
        
        current_price = self.data['Close'].iloc[-1]
        predictions = []
        
        for day in range(1, days + 1):
            pred = best_model.predict(last_features_scaled)[0]
            change_pct = ((pred - current_price) / current_price) * 100
            predictions.append(pred)
            
            print(f"Day {day}: ${pred:.2f} ({change_pct:+.2f}%)")
        
        return predictions
    
    def display_summary(self):
        """Display summary statistics"""
        if self.data is None:
            print("❌ No data available.")
            return
        
        current_price = self.data['Close'].iloc[-1]
        price_change = self.data['Close'].iloc[-1] - self.data['Close'].iloc[-2]
        price_change_pct = (price_change / self.data['Close'].iloc[-2]) * 100
        
        print(f"\n📊 {self.ticker} Summary:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Daily Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")
        print(f"52W High: ${self.data['High'].max():.2f}")
        print(f"52W Low: ${self.data['Low'].min():.2f}")
        print(f"Average Volume: {self.data['Volume'].mean():,.0f}")
        
        if self.models:
            print(f"\n🤖 Model Performance:")
            for name, results in self.models.items():
                print(f"{name}: R² = {results['r2']:.4f}, RMSE = ${results['rmse']:.2f}")

def main():
    """Main function to run the stock predictor"""
    print("🚀 Advanced Stock Price Predictor")
    print("=" * 40)
    
    # Get user input
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    
    if not ticker:
        ticker = "AAPL"
        print("Using default ticker: AAPL")
    
    # Initialize predictor
    predictor = StockPredictor(ticker)
    
    # Fetch and process data
    if predictor.fetch_data():
        predictor.calculate_technical_indicators()
        predictor.display_summary()
        
        # Train models
        print("\n🤖 Training machine learning models...")
        predictor.train_models()
        
        # Make future predictions
        predictor.predict_future(days=7)
        
        # Plot results
        print("\n📈 Generating plots...")
        predictor.plot_results()
    
    else:
        print("Failed to fetch data. Please check the ticker symbol.")

if __name__ == "__main__":
    main()
