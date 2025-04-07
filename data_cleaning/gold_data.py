import pandas as pd
from datetime import date
import yfinance as yf
from fredapi import Fred

today = date.today()

gold_ticker = 'GC=F'
silver_ticker = 'SI=F'
usd_index = 'DX-Y.NYB'
oil_ticker = 'WTI'
GPR_ticker = 'GPR.AX'
stock_market_volatility = '^VIX'

ticker_dict = {'gold_price':'GC=F', 'silver_price':'SI=F', 'platinum_price': 'PL=F', 'palladium_price':'PA=F', 'stock_violatility':'^VIX', 'usd_strength':'DX-Y.NYB', 'oil':'WTI', 'geo_resources':'GPR.AX'}

df = pd.DataFrame()
for key, value in ticker_dict.items():
    df[key] = yf.download(value, start="2015-01-01", end=today)['Close']


df["gold_price_change"] = df["gold_price"].pct_change().fillna(0)
df["silver_price_change"] = df["silver_price"].pct_change().fillna(0)
df["platinum_price_change"] = df["platinum_price"].pct_change().fillna(0)
df["palladium_price_change"] = df["palladium_price"].pct_change().fillna(0)

df["gold_lag1"] = df["gold_price_change"].shift(1).fillna(0)  # 1-day lag
df["gold_lag7"] = df["gold_price_change"].shift(7).fillna(0)  # 1-week lag
df["gold_lag14"] = df["gold_price_change"].shift(14).fillna(0) # 2-week lag

df["gold_7d_avg"] = df["gold_price"].rolling(window=7).mean().fillna(0)
df["gold_14d_avg"] = df["gold_price_change"].rolling(window=14).mean().fillna(0)
df["gold_30d_avg"] = df["gold_price"].rolling(window=30).mean().fillna(0)

df["gold_vol_7d"] = df["gold_price_change"].rolling(7).std().fillna(0)
df["gold_vol_30d"] = df["gold_price_change"].rolling(30).std().fillna(0)


# Set your FRED API key
fred = Fred(api_key="332b9ccf07ba9723df4010a46394565f")  # <-- Replace this with your actual key

# Date range
start_date = "2015-01-01"
end_date = today

macro_series = {
    "fed_funds_rate": "FEDFUNDS",
    "inflation": "CPIAUCSL",          # Monthly
    "GDP_growth": "A191RL1Q225SBEA",  # Quarterly
    "unemployment": "UNRATE"          # Monthly
}

macro_data = pd.DataFrame()

for name, series_id in macro_series.items():
    print(f"Fetching macro data for {name} ({series_id})...")
    series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df_r = pd.DataFrame(series, columns=[name])
    df_r.index.name = "Date"
    macro_data = macro_data.join(df_r, how='outer') if not macro_data.empty else df_r

# Daily resampling with forward fill
macro_data.index = pd.to_datetime(macro_data.index)
macro_daily = macro_data.resample("D").ffill()


df = pd.merge(df, macro_daily, on='Date')
df['gdp_update_flag'] = df['GDP_growth'].notna().astype(int)
df['GDP_growth'] = df['GDP_growth'].ffill()

final_data = df
print(final_data.head())

final_data.to_csv('data/gold_data.csv', index=True)

