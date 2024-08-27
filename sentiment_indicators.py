from urllib.request import urlopen, Request;
from bs4 import BeautifulSoup;
from nltk.sentiment.vader import SentimentIntensityAnalyzer;
import pandas as pd;
from datetime import datetime, timedelta;
import matplotlib.pyplot as plt;

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = [ 'AAPL','AMZN', 'GOOG', 'META']

news_tables = {}

for ticker in tickers:
    url= finviz_url + ticker

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

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
vader = SentimentIntensityAnalyzer()

df['compound'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])
print(df['compound'].describe())
df['date']= pd.to_datetime(df.date).dt.date
plt.figure(figsize=(10,8))
mean_df = df.groupby(['ticker', 'date'])['compound'].mean().unstack()
print(mean_df.describe())
mean_df = mean_df.transpose()
mean_df.plot(kind='bar')
plt.show()