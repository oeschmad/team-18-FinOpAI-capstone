# predict_stock.py
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from datetime import datetime
# Constants
STOCK_CATEGORIES = {
    'Technology': ['AAPL', 'MSFT', 'NVDA'],
    'Financials': ['JPM', 'BAC', 'WFC'],
    'Healthcare': ['JNJ', 'LLY', 'PFE'],
    'Consumer Discretionary': ['A', 'TSLA', 'HD'],
    'Consumer Staples': ['PG', 'KO', 'PEP'],
    'Industrials': ['CAT', 'BA', 'DE'],
    'Energy': ['XOM', 'CVX', 'SLB'],
    'Utilities': ['DUK', 'SO', 'NEE'],
    'Materials': ['LIN', 'SHW', 'FCX'],
    'Communications': ['GOOGL', 'META', 'DIS']
}
SEQ_LENGTH = 60
HORIZON_TO_DAYS = {
    "1 Day": 1,
    "5 Days": 5,
    "1 Week": 7,
    "2 Weeks": 14,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}
def load_stock_data(path='raw_stock_data.xlsx'):
    """Load stock data and handle column names with _Close suffix"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing stock data at {path}")
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()  # Normalize column names
    if 'Date' not in df.columns:
        raise ValueError("Missing 'Date' column in dataset.")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.set_index('Date')
    # Map columns to remove _Close suffix for processing
    col_mapping = {col: col.replace('_Close', '') for col in df.columns if '_Close' in col}
    df = df.rename(columns=col_mapping)
    print(":white_check_mark: Loaded columns:", df.columns.tolist())
    print(df.head())
    return df.fillna(method='ffill')
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True),  # Slightly smaller LSTM layer
        Dropout(0.15),  # Reduced dropout
        LSTM(32),      # Slightly smaller LSTM layer
        Dropout(0.15),  # Reduced dropout
        Dense(16),      # Smaller dense layer
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') # Adjusted learning rate
    return model
def forecast_stock_returns(investment_horizon="30", data_path='raw_stock_data.xlsx', model_dir='models'):
    if isinstance(investment_horizon, int):
        n_future_days = investment_horizon
    elif investment_horizon in HORIZON_TO_DAYS:
        n_future_days = HORIZON_TO_DAYS[investment_horizon]
    else:
        raise ValueError(f"Unknown investment horizon: {investment_horizon}")
    os.makedirs(model_dir, exist_ok=True)
    df = load_stock_data(data_path)
    # Ensure data extends to today (using today's date in Bengaluru)
    today = pd.Timestamp('2025-04-20').normalize()
    last_date = df.index[-1]
    if last_date < today:
        missing_days = (today - last_date).days
        pad = pd.DataFrame([df.loc[last_date]] * missing_days,
                          index=pd.date_range(last_date + pd.Timedelta(days=1), today))
        df = pd.concat([df, pad])
    forecast_df = pd.DataFrame()
    for category, stocks in STOCK_CATEGORIES.items():
        print(f"Processing {category} stocks...")
        for stock in stocks:
            if stock not in df.columns:
                print(f":warning: Skipping {stock}: not found in data columns.")
                continue
            series = df[[stock]].dropna()
            if len(series) < SEQ_LENGTH + n_future_days:
                print(f":warning: Skipping {stock}: not enough data (needs {SEQ_LENGTH + n_future_days}, has {len(series)})")
                continue
            scaler = MinMaxScaler()
            scaled_prices = scaler.fit_transform(series)
            # === Prepare sequences for return prediction ===
            X, y = [], []
            for i in range(SEQ_LENGTH, len(scaled_prices) - 1):
                X.append(scaled_prices[i - SEQ_LENGTH:i])
                # Predict % return from price i to i+1
                pct_return = (scaled_prices[i + 1][0] - scaled_prices[i][0]) / max(scaled_prices[i][0], 0.001) # Avoid division by zero
                y.append([pct_return])
            X, y = np.array(X), np.array(y)
            model = build_lstm_model((SEQ_LENGTH, 1))
            model.fit(X, y, epochs=2, batch_size=64, verbose=0) # Reduced epochs, increased batch size
            # Forecast % returns
            last_seq = scaled_prices[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
            last_price = scaled_prices[-1][0]
            future_scaled_prices = [last_price]
            predicted_returns = []
            for _ in range(n_future_days - 1):
                next_return = model.predict(last_seq, verbose=0)[0][0]
                predicted_returns.append(next_return)
                # Update next price using predicted return
                next_price = future_scaled_prices[-1] * (1 + next_return)
                future_scaled_prices.append(next_price)
                # Update input sequence
                last_seq = np.append(last_seq[:, 1:, :], [[[next_price]]], axis=1)
            # Convert scaled prices back to actual prices
            scaled_prices_array = np.array(future_scaled_prices).reshape(-1, 1)
            real_prices = scaler.inverse_transform(scaled_prices_array).flatten()
            # Calculate real % returns
            actual_returns = np.diff(real_prices) / real_prices[:-1]
            start_date = series.index[-1] + pd.Timedelta(days=1)
            forecast_dates = pd.date_range(start=start_date, periods=n_future_days - 1)
            forecast_df[stock] = pd.Series(data=actual_returns, index=forecast_dates)
            # Save the model
            model.save(os.path.join(model_dir, f"{stock}_lstm_model.keras"))
    forecast_df.index.name = 'Date'
    print(f"[INFO] Generated stock return forecast for next {n_future_days - 1} days.")
    print(forecast_df.head())
    forecast_df.to_csv('stock_forecasts.csv')
    print(":white_check_mark: Saved forecasts to stock_forecasts.csv")
    return forecast_df
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_stock.py '<Time Horizon>'")
        print("Options:", list(HORIZON_TO_DAYS.keys()) + ['<Integer (days)>'])
        sys.exit(1)
    user_horizon = sys.argv[1]
    try:
        user_horizon = int(user_horizon)
    except ValueError:
        pass  # keep string
    forecast_stock_returns(investment_horizon=user_horizon)
