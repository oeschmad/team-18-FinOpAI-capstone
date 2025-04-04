import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

import yfinance as yf
from fredapi import Fred

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv('data/gold_data.csv')
today = date.today()

#Define features and target
features = ['inflation', 'stock_violatility', 'usd_strength', 'gold_lag1', 'gold_lag7', 'gold_lag14', 'gold_7d_avg', 'gold_14d_avg',
            'gold_30d_avg', 'oil', 'geo_resources', 'gold_vol_7d','gold_vol_30d', 'fed_funds_rate', 'GDP_growth',
            'unemployment', 'silver_price_change', 'platinum_price_change','palladium_price_change']

target = "gold_price_change"

def gold_model(features, target):
    """this function takes 2 inputs, the features adn the target, and returns the best XGBoost Model"""

    X = df[features]
    y = df[target]

    # Train-test split
    train_size = int(0.7 * len(df))  # 70% train
    val_size = int(0.15 * len(df))  # 15% validation
    test_size = len(df) - train_size  # Remaining 15% test

    # Split data chronologically
    train = df.iloc[:train_size]
    val = df.iloc[train_size:train_size+val_size]
    test = df.iloc[train_size+val_size:]

    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

    # Train Initial XGBoost model
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
    xgb_model.fit(X_train, y_train)

    #Feature Importance
    importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
    print("Feature Importances:\n", importances)

    #Drop Least Important Features
    drop_candidates = importances[importances < 0.02].index.tolist()  # Drop features with very low importance

    for drop_feature in drop_candidates:
        X_train_reduced = X_train.drop(columns=[drop_feature])
        X_test_reduced = X_test.drop(columns=[drop_feature])

    xgb_model_reduced = xgb.XGBRegressor(objective="reg:squarederror")
    xgb_model_reduced.fit(X_train_reduced, y_train)

    #Tune Hyperparameters
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    model = xgb.XGBRegressor(objective='reg:squarederror')

    random_search = RandomizedSearchCV(
        estimator=model, param_distributions=param_grid, n_iter=20,
        scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1
    )

    random_search.fit(X_train_reduced, y_train)

    best_model = random_search.best_estimator_

    best_params = random_search.best_params_
    print('best_params : ', best_params)

    y_pred = best_model.predict(X_test_reduced)
    print(f"R-squared: {r2_score(y_test, y_pred):.4f}")
    print("Mean squared error: %.7f" % mean_squared_error(y_test, y_pred))

    return best_model

gold_model(features, target)
