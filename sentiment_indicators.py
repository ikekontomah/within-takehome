import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
# 1. Add Technical Indicators (RSI, MACD, Bollinger)
###############################################################################
def add_technical_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_period=20):
    """
    Compute RSI, MACD, and Bollinger Bands, and add them as new columns.
    """
    # Ensure 'Close' exists
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' price column to compute indicators.")

    # ====================
    # RSI
    # ====================
    close = df['Close']
    delta = close.diff()

    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ema_up = up.ewm(span=rsi_period, adjust=False).mean()
    ema_down = down.ewm(span=rsi_period, adjust=False).mean()

    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # ====================
    # MACD
    # ====================
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ====================
    # Bollinger Bands
    # ====================
    df['BB_middle'] = close.rolling(window=bb_period).mean()
    df['BB_std'] = close.rolling(window=bb_period).std()

    df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
    df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])

    # Drop the 'BB_std' helper if you prefer
    df.drop(columns=['BB_std'], inplace=True)

    return df

###############################################################################
# 2. Sentiment Analyzers
###############################################################################
vader_analyzer = SentimentIntensityAnalyzer()
transformer_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased")

FIN_POSITIVE_WORDS = {"profit", "growth", "gain", "bullish", "positive", "outperform"}
FIN_NEGATIVE_WORDS = {"loss", "decline", "risk", "bearish", "negative", "underperform"}

def analyze_sentiment_lm(text):
    words = text.lower().split()
    pos_count = sum(word in FIN_POSITIVE_WORDS for word in words)
    neg_count = sum(word in FIN_NEGATIVE_WORDS for word in words)
    return float(pos_count - neg_count)

def analyze_sentiment_textblob(text):
    return TextBlob(text).sentiment.polarity

def analyze_sentiment_transformers(text):
    result = transformer_sentiment(text)
    score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
    return score

def analyze_sentiment_vader(text):
    return vader_analyzer.polarity_scores(text)['compound']

def analyze_sentiment(text, method='vader'):
    if method == 'vader':
        return analyze_sentiment_vader(text)
    elif method == 'transformer':
        return analyze_sentiment_transformers(text)
    elif method == 'textblob':
        return analyze_sentiment_textblob(text)
    elif method == 'lm':
        return analyze_sentiment_lm(text)
    else:
        raise ValueError("Unsupported sentiment method. Choose from: 'vader', 'transformer', 'textblob', or 'lm'.")

def process_sentiment(df, text_col='Text', method='vader'):
    df['Sentiment'] = df[text_col].apply(lambda x: analyze_sentiment(str(x), method))
    return df

###############################################################################
# 3. Data Fetching
###############################################################################
def fetch_financial_news(api_key, query="NFLX", days_ago=30, language="en", page_size=100):
    today = dt.date.today()
    from_date = (today - dt.timedelta(days=days_ago)).isoformat()
    to_date = today.isoformat()

    newsapi = NewsApiClient(api_key=api_key)
    data = newsapi.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language=language,
        sort_by="relevancy",
        page_size=page_size
    )

    articles_all = []
    if "articles" in data:
        for article in data["articles"]:
            articles_all.append({
                "source": article["source"]["name"],
                "author": article["author"],
                "title": article["title"],
                "description": article["description"],
                "url": article["url"],
                "publishedAt": article["publishedAt"],
                "content": article["content"]
            })

    return pd.DataFrame(articles_all)

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(1)

    # Compute returns from 'Close'
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data['Target'] = (stock_data['Return'] > 0).astype(int)
    return stock_data

def merge_data(stock_data, sentiment_data):
    stock_data = stock_data.reset_index()
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.date

    merged_data = pd.merge(stock_data, sentiment_data, on='Date', how='left')
    merged_data['Sentiment'].fillna(0, inplace=True)
    return merged_data

###############################################################################
# 4. Model Training (Incorporate new technical indicators)
###############################################################################
def train_random_forest(data):
    # Include new indicator columns in the features
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0)

    target = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                        test_size=0.2, 
                                                        random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Random Forest ---")
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy Score:", acc)

    return model, acc

def train_logistic_regression(data):
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0)

    target = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2, 
                                                        random_state=42)
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy Score:", acc)
    return model, acc

def train_naive_bayes(data):
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0)

    target = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2, 
                                                        random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Naive Bayes ---")
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy Score:", acc)
    return model, acc

def train_rnn(data):
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0).values
    target = data['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                        test_size=0.2,
                                                        random_state=42)
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn  = np.reshape(X_test,  (X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(16, input_shape=(1, X_train.shape[1])))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X_test_rnn, y_test, verbose=0)
    y_pred = (model.predict(X_test_rnn) > 0.5).astype(int).flatten()

    print("\n--- LSTM RNN ---")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score: {accuracy:.3f}")
    return model, accuracy

def train_cnn(data):
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0).values
    target = data['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        random_state=42)
    X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn  = np.reshape(X_test,  (X_test.shape[0],  X_test.shape[1], 1))

    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=2, activation='relu', input_shape=(X_train.shape[1],1)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
    y_pred = (model.predict(X_test_cnn) > 0.5).astype(int).flatten()

    print("\n--- 1D CNN ---")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score: {accuracy:.3f}")
    return model, accuracy

###############################################################################
# 5. Visualization Helpers
###############################################################################
def plot_stock_and_sentiment(merged_df, stock_label="Stock & Sentiment Over Time"):
    fig, ax1 = plt.subplots(figsize=(10,6))
    merged_df = merged_df.sort_values(by='Date')

    dates = merged_df['Date']
    close_prices = merged_df['Close']
    sentiment = merged_df['Sentiment']

    ax1.plot(dates, close_prices, color='blue', label='Close Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(dates, sentiment, color='red', label='Sentiment')
    ax2.set_ylabel('Sentiment', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(stock_label)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.show()

def plot_accuracy_scores(acc_dict):
    # Convert dictionary into a DataFrame
    df_plot = pd.DataFrame({
        'Classifier': list(acc_dict.keys()),
        'Accuracy': list(acc_dict.values())
    })

    plt.figure(figsize=(8,5))
    sns.barplot(
        data=df_plot,
        x='Classifier',
        y='Accuracy',
        hue='Classifier',   # ensures each bar has a unique color
        dodge=False,
        palette='viridis',
        legend=False
    )
    plt.ylim(0,1)
    plt.title("Comparison of Classifier Accuracy")
    plt.ylabel("Accuracy")

    # Optional: Annotate bar values
    for i, row in df_plot.iterrows():
        plt.text(i, row['Accuracy']+0.01, f"{row['Accuracy']:.2f}", ha='center', fontweight='bold')

    plt.show()


def plot_sentiment_distribution(df, stock_label="GOOG"):
    plt.figure(figsize=(8,5))
    sns.histplot(df['Sentiment'], kde=True, color='darkgreen')
    plt.title(f"Sentiment Distribution for {stock_label}")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

###############################################################################
# 6. Main Execution Example
###############################################################################
if __name__ == "__main__":
    # 1) Fetch News
    api_key = "0341e1c1300a4e62b0fcdd849984821c"  
    news_df = fetch_financial_news(api_key=api_key, query="NVDA", days_ago=30)
    if not news_df.empty:
        news_df['Text'] = news_df['title'].astype(str) + " " + news_df['description'].astype(str)
        news_df = process_sentiment(news_df, text_col='Text', method='transformer')
        news_df['Date'] = pd.to_datetime(news_df['publishedAt']).dt.date
        daily_sent = news_df.groupby('Date')['Sentiment'].mean().reset_index()
    else:
        daily_sent = pd.DataFrame(columns=['Date','Sentiment'])

    # 2) Fetch Stock Data, then add RSI/MACD/Bollinger
    ticker = "NVDA"
    start_date = "2024-12-01"
    end_date   = "2025-01-26"
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data = add_technical_indicators(stock_data)  # <-- new indicators

    # 3) Merge
    merged_data = merge_data(stock_data, daily_sent)

    # (Optional) Show top rows
    print("\nMerged DataFrame (Stock + News Sentiment + Indicators):")
    print(merged_data[['Date','Close','Sentiment','RSI','MACD','MACD_signal','BB_upper','BB_lower']].head(10))

    # 4) Visualize
    plot_stock_and_sentiment(merged_data, stock_label=f"{ticker} Stock & Sentiment")

    # 5) Train models & compare
    acc_results = {}
    rf_model, rf_acc = train_random_forest(merged_data)
    acc_results["RandomForest"] = rf_acc

    lr_model, lr_acc = train_logistic_regression(merged_data)
    acc_results["LogisticRegression"] = lr_acc

    nb_model, nb_acc = train_naive_bayes(merged_data)
    acc_results["NaiveBayes"] = nb_acc

    rnn_model, rnn_acc = train_rnn(merged_data)
    acc_results["RNN"] = rnn_acc

    cnn_model, cnn_acc = train_cnn(merged_data)
    acc_results["CNN"] = cnn_acc

    # Plot accuracy comparison
    plt.show()
    plot_accuracy_scores(acc_results)
    plt.show()

    # Show sentiment distribution
    plt.show()
    plot_sentiment_distribution(merged_data, stock_label=ticker)
    plt.show()

    print("\n Combined sentiment + RSI/MACD/Bollinger pipeline executed.")
