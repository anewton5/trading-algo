import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

allowed_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

def is_valid_ticker(ticker):
    return ticker.isalnum() and 1 <= len(ticker) <= 5

def is_valid_period(period):
    return period in allowed_periods

while True:
    stock_symbol = input("Enter the stock ticker symbol: ").upper()
    if is_valid_ticker(stock_symbol):
        df_test = yf.download(stock_symbol, period='1d')
        if not df_test.empty:
            break
        else:
            print("Ticker symbol not found or no data available. Please enter a valid stock ticker (e.g., AAPL).")
    else:
        print("Invalid ticker symbol. Please use a valid stock ticker in correct form (e.g., AAPL).")

while True:
    print("Enter the lookback period for the moving average and z-score calculation (must be one of the following):")
    for period in allowed_periods:
        print(f"  - {period}")
    lookback_period = input()
    if is_valid_period(lookback_period):
        break
    else:
        print(f"Invalid period. Please choose one of the listed periods.")

df_stock = yf.download(stock_symbol, period=lookback_period)
df_etf = yf.download('SPY', period=lookback_period)

if 'mo' in lookback_period:
    lookback_weeks = int(lookback_period[:-2]) * 4
elif 'y' in lookback_period:
    lookback_weeks = int(lookback_period[:-1]) * 52
else:
    lookback_weeks = int(lookback_period[:-1])

df_stock["ema"] = df_stock["Close"].ewm(span=lookback_weeks, adjust=False).mean()
df_etf["ema"] = df_etf["Close"].ewm(span=lookback_weeks, adjust=False).mean()

df_stock["z_score"] = (df_stock["Close"] - df_stock["ema"]) / df_stock["Close"].ewm(span=lookback_weeks, adjust=False).std()
df_etf["z_score"] = (df_etf["Close"] - df_etf["ema"]) / df_etf["Close"].ewm(span=lookback_weeks, adjust=False).std()

def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_stock["rsi"] = calculate_rsi(df_stock["Close"], 14)
df_etf["rsi"] = calculate_rsi(df_etf["Close"], 14)

df_stock = df_stock.reset_index()
df_etf = df_etf.reset_index()
df = pd.merge(df_stock, df_etf, on="Date", suffixes=('_stock', '_etf'))

z_score_threshold = 1.5
rsi_overbought = 70
rsi_oversold = 30

def determine_outperformance(row):
    if row["z_score_stock"] > z_score_threshold and row["rsi_stock"] > rsi_overbought:
        return "Strong Overperforming"
    elif row["z_score_stock"] > 0 and row["rsi_stock"] > 50:
        return "Overperforming"
    elif row["z_score_stock"] < -z_score_threshold and row["rsi_stock"] < rsi_oversold:
        return "Strong Underperforming"
    elif row["z_score_stock"] < 0 and row["rsi_stock"] < 50:
        return "Underperforming"
    else:
        return "In line with"

df["outperformance"] = df.apply(determine_outperformance, axis=1)

plt.figure(figsize=(14, 7))
plt.plot(df["Date"], df["Close_stock"], label=f'{stock_symbol} Price', color='blue')
plt.plot(df["Date"], df["ema_stock"], label=f'{stock_symbol} EMA', color='orange')

underperforming = df[df["outperformance"] == "Underperforming"]
plt.scatter(underperforming["Date"], underperforming["Close_stock"], color='red', label='Underperforming', marker='v')

strong_underperforming = df[df["outperformance"] == "Strong Underperforming"]
plt.scatter(strong_underperforming["Date"], strong_underperforming["Close_stock"], color='darkred', label='Strong Underperforming', marker='v')

overperforming = df[df["outperformance"] == "Overperforming"]
plt.scatter(overperforming["Date"], overperforming["Close_stock"], color='limegreen', label='Overperforming', marker='^')

strong_overperforming = df[df["outperformance"] == "Strong Overperforming"]
plt.scatter(strong_overperforming["Date"], strong_overperforming["Close_stock"], color='darkgreen', label='Strong Overperforming', marker='^')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{stock_symbol} Price and EMA with Performance Indicators')
plt.legend()
plt.grid(True)
plt.show()