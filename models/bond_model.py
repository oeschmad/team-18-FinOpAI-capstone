import pandas as pd
import numpy as np
import joblib
import os
import sys
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Constants
BOND_COLS = [
    '10-Year US Treasury', '5-Year US Treasury', '30-Year US Treasury',
    'Corporate Bonds (LQD)', 'High-Yield Bonds (HYG)', 'Municipal Bonds (MUB)'
]
MACRO_COLS = ['Unemployment', 'Inflation', 'Fed Funds Rate']
LAGS = [1, 3, 6, 12]

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
    for col in BOND_COLS:
        for lag in LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        df[f"{col}_rolling_3"] = df[col].rolling(3).mean()
        df[f"{col}_rolling_6"] = df[col].rolling(6).mean()

    df["10y-2y_Spread"] = df["10-Year US Treasury"] - df["5-Year US Treasury"]
    df["30y-10y_Spread"] = df["30-Year US Treasury"] - df["10-Year US Treasury"]
    df["Unemp*Inflation"] = df["Unemployment"] * df["Inflation"]
    return df

def prepare_model_inputs(df):
    feature_cols = (
        [f"{col}_lag_{lag}" for col in BOND_COLS for lag in LAGS] +
        [f"{col}_rolling_3" for col in BOND_COLS] +
        [f"{col}_rolling_6" for col in BOND_COLS] +
        ["10y-2y_Spread", "30y-10y_Spread", "Unemp*Inflation"] +
        MACRO_COLS
    )
    return df[feature_cols]

def predict_bonds(investment_horizon="3 Months",
                  data_path='bond_final_data.csv',
                  model_dir='models'):
    """
    Trains on all historical data and predicts future bond yields
    based on the investment horizon. Returns a forecast DataFrame.
    """
    # Allow integer horizon as days directly
    if isinstance(investment_horizon, int):
        n_future_days = investment_horizon
    elif investment_horizon in HORIZON_TO_DAYS:
        n_future_days = HORIZON_TO_DAYS[investment_horizon]
    else:
        raise ValueError(f"Unknown investment horizon: {investment_horizon}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No input data found at {data_path}")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.set_index('Date')

    # Pad forward if data is outdated
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
    y = df[BOND_COLS]

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

    forecast_rows = []
    current_df = df.copy()
    last_known_index = df.index[-1]

    for _ in range(n_future_days):
        next_date = last_known_index + pd.Timedelta(days=1)
        last_known_index = next_date

        latest = current_df.iloc[-1:].copy()

        next_row = pd.DataFrame(index=[next_date])
        for col in MACRO_COLS:
            next_row[col] = latest[col].values[0]

        for col in BOND_COLS:
            next_row[col] = np.nan

        current_df = pd.concat([current_df, next_row])
        current_df = create_features(current_df)
        current_df.dropna(inplace=True)

        X_next = prepare_model_inputs(current_df.iloc[[-1]])
        y_pred = model.predict(X_next)[0]

        for i, col in enumerate(BOND_COLS):
            current_df.at[next_date, col] = y_pred[i]

        forecast_rows.append((next_date, y_pred))

    forecast_df = pd.DataFrame(
        [vals for _, vals in forecast_rows],
        columns=BOND_COLS,
        index=[d for d, _ in forecast_rows]
    )
    forecast_df.index.name = 'Date'

    joblib.dump(model, os.path.join(model_dir, "bond_model_future.pkl"))
    print(f"[INFO] Generated bond forecast for next {n_future_days} days.")
    print(forecast_df.head())
    return forecast_df

