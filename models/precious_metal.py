import pandas as pd
import numpy as np
import os
import sys
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Constants
METAL_COLS = ['gold_price_change', 'silver_price_change', 'platinum_price_change', 'palladium_price_change']
MACRO_COLS = ['inflation', 'unemployment', 'fed_funds_rate', 'usd_strength', 'oil']
LAGS = [1, 3, 7, 14]

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

def create_features(df):
    df = df.copy()
    for col in METAL_COLS:
        for lag in LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        df[f"{col}_rolling_3"] = df[col].rolling(3).mean()
        df[f"{col}_rolling_7"] = df[col].rolling(7).mean()
    df["gold_silver_spread"] = df["gold_price_change"] - df["silver_price_change"]
    df["oil_gold_interaction"] = df["oil"] * df["gold_price_change"]
    return df

def prepare_model_inputs(df):
    feature_cols = (
        [f"{col}_lag_{lag}" for col in METAL_COLS for lag in LAGS] +
        [f"{col}_rolling_3" for col in METAL_COLS] +
        [f"{col}_rolling_7" for col in METAL_COLS] +
        ["gold_silver_spread", "oil_gold_interaction"] +
        MACRO_COLS
    )
    return df[feature_cols]


def predict_precious_metals(investment_horizon="3 Months",
                             data_path='gold_data.csv',
                             model_dir='models'):
    if isinstance(investment_horizon, str):
        if investment_horizon not in HORIZON_TO_DAYS:
            raise ValueError(f"Unknown investment horizon: {investment_horizon}")
        n_future_days = HORIZON_TO_DAYS[investment_horizon]
    elif isinstance(investment_horizon, int):
        n_future_days = investment_horizon
    else:
        raise ValueError("investment_horizon must be either a string or an integer.")

    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.set_index('Date')

    today = pd.Timestamp.today().normalize()
    last_date = df.index[-1]
    if last_date < today:
        missing_days = (today - last_date).days
        pad = pd.DataFrame([df.loc[last_date]] * missing_days,
                           index=pd.date_range(last_date + pd.Timedelta(days=1), today))
        df = pd.concat([df, pad])

    df = create_features(df)
    df.dropna(inplace=True)

    X = prepare_model_inputs(df)
    y = df[METAL_COLS]

    model = MultiOutputRegressor(XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.06,
        max_depth=3,
        colsample_bytree=0.9,
        subsample=0.85,
        reg_lambda=2,
        reg_alpha=1,
        verbosity=0
    ))
    model.fit(X, y)

    # Forecasting loop
    forecast_rows = []
    current_df = df.copy()
    last_known_index = current_df.index[-1]

    for _ in range(n_future_days):
        next_date = last_known_index + pd.Timedelta(days=1)
        last_known_index = next_date

        latest = current_df.iloc[-1:].copy()

        next_row = pd.DataFrame(index=[next_date])
        for col in MACRO_COLS:
            next_row[col] = latest[col].values[0]
        for col in METAL_COLS:
            next_row[col] = np.nan

        current_df = pd.concat([current_df, next_row])
        current_df = create_features(current_df)
        current_df.dropna(inplace=True)

        X_next = prepare_model_inputs(current_df.iloc[[-1]])
        y_pred = model.predict(X_next)[0]

        for i, col in enumerate(METAL_COLS):
            current_df.at[next_date, col] = y_pred[i]

        forecast_rows.append((next_date, y_pred))

    forecast_df = pd.DataFrame(
        [vals for _, vals in forecast_rows],
        columns=METAL_COLS,
        index=[d for d, _ in forecast_rows]
    )
    forecast_df.index.name = 'Date'

    joblib.dump(model, os.path.join(model_dir, "precious_metals_model_future.pkl"))
    print(f"[INFO] Generated precious metals forecast for next {n_future_days} days.")
    print(forecast_df.head())
    return forecast_df
