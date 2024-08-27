# trading-algo

Trading algorithm, combining sentiment analysis with EMA and momentum performance indicators. Using beautifulsoup4 to scrape news headlines and yfinance to fetch stock data, the algorithm performs the following steps:

1. **Input Validation**: Validates the stock ticker symbol and lookback period.
2. **Data Fetching**: Downloads historical stock data and ETF data for the specified period using yfinance.
3. **EMA Calculation**: Computes the Exponential Moving Average (EMA) for both the stock and ETF.
4. **Sentiment Analysis**: Scrapes news headlines from Finviz, performs sentiment analysis using NLTK's VADER, and calculates the average sentiment score.
5. **Z-Score and RSI Calculation**: Computes the Z-Score and Relative Strength Index (RSI) for the stock.
6. **Performance Classification**: Classifies the stock's performance based on Z-Score and RSI thresholds.
7. **Data Visualization**: Plots the stock price, EMA, and performance indicators on a graph.

The algorithm provides a comprehensive view of the stock's performance by integrating technical indicators with sentiment analysis.