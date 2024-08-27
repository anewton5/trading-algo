import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

# Sentiment Analysis
finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = [stock_symbol]

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'User-Agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.strip().split(' ')
        if len(date_data) == 1:
            time = date_data[0].strip()
        else:
            date = date_data[0].strip()
            time = date_data[1].strip()
            if date == 'Today':
                date = datetime.now().date()
            elif date == 'Yesterday':
                date = datetime.now().date() - timedelta(1)
            else:
                date = pd.to_datetime(date).date()
        
        parsed_data.append([ticker, date, time, title])

df_news = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
vader = SentimentIntensityAnalyzer()

df_news['compound'] = df_news['title'].apply(lambda title: vader.polarity_scores(title)['compound'])
df_news['date'] = pd.to_datetime(df_news.date).dt.date

mean_df = df_news.groupby(['ticker', 'date'])['compound'].mean().unstack()

print("mean_df:")
print(mean_df.head())

mean_df_long = mean_df.stack().reset_index()
mean_df_long.columns = ['ticker', 'date', 'compound']

print("mean_df_long:")
print(mean_df_long.head())

df_stock['date'] = df_stock.index.date
df_stock = df_stock.merge(mean_df_long[['date', 'compound']], on='date', how='left')

print("df_stock after merge:")
print(df_stock.head())

df_stock['positive_sentiment'] = df_stock['compound'].apply(lambda x: x if x > 0 else 0)
df_stock['negative_sentiment'] = df_stock['compound'].apply(lambda x: x if x < 0 else 0)

rolling_mean = df_stock['Close'].rolling(window=lookback_weeks).mean()
rolling_std = df_stock['Close'].rolling(window=lookback_weeks).std()
df_stock['z_score_stock'] = (df_stock['Close'] - rolling_mean) / rolling_std

print("df_stock with z_score_stock:")
print(df_stock[['Close', 'z_score_stock']].head())

def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_stock['rsi_stock'] = calculate_rsi(df_stock['Close'], window=14)

print("df_stock with rsi_stock:")
print(df_stock[['Close', 'rsi_stock']].head())

def determine_outperformance(row):
    z_score_threshold = 1.0  
    rsi_overbought = 70  
    rsi_oversold = 30 

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

df_stock["outperformance"] = df_stock.apply(determine_outperformance, axis=1)

df_stock.fillna({'compound': 0}, inplace=True)

df_stock.dropna(subset=['z_score_stock', 'rsi_stock'], inplace=True)

print("df_stock after handling NaN values:")
print(df_stock.head())

print(df_stock.columns)

if 'date' in df_stock.columns:
    df_stock.rename(columns={'date': 'Date'}, inplace=True)

print(df_stock.head())

plt.figure(figsize=(14, 7))
plt.plot(df_stock["Date"], df_stock["Close"], label=f'{stock_symbol} Price', color='blue')
plt.plot(df_stock["Date"], df_stock["ema"], label=f'{stock_symbol} EMA', color='orange')

underperforming = df_stock[df_stock["outperformance"] == "Underperforming"]
plt.scatter(underperforming["Date"], underperforming["Close"], color='red', label='Underperforming', marker='v')

strong_underperforming = df_stock[df_stock["outperformance"] == "Strong Underperforming"]
plt.scatter(strong_underperforming["Date"], strong_underperforming["Close"], color='darkred', label='Strong Underperforming', marker='v')

overperforming = df_stock[df_stock["outperformance"] == "Overperforming"]
plt.scatter(overperforming["Date"], overperforming["Close"], color='limegreen', label='Overperforming', marker='^')

strong_overperforming = df_stock[df_stock["outperformance"] == "Strong Overperforming"]
plt.scatter(strong_overperforming["Date"], strong_overperforming["Close"], color='darkgreen', label='Strong Overperforming', marker='^')

positive_sentiment = df_stock[df_stock["positive_sentiment"] > 0]
plt.scatter(positive_sentiment["Date"], positive_sentiment["Close"], color='cyan', label='Positive Sentiment', marker='o')

negative_sentiment = df_stock[df_stock["negative_sentiment"] < 0]
plt.scatter(negative_sentiment["Date"], negative_sentiment["Close"], color='magenta', label='Negative Sentiment', marker='x')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{stock_symbol} Price and EMA with Performance and Sentiment Indicators')
plt.legend()
plt.grid(True)
plt.show()