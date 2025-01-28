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


# Initialize Sentiment Analyzers

vader_analyzer = SentimentIntensityAnalyzer()
distilbert_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased")

finbert_sentiment = pipeline(
    "sentiment-analysis", 
    model="ProsusAI/finbert", 
    tokenizer="ProsusAI/finbert"
)

# 1. Fetch Financial News for real world sentiments
''' Stocks being considered and their tickers:  = Apple: AAPL, Nvidia: NVDA, Amazon: AMZN, Qualcomm: QCOM, 
Google: GOOG, Microsoft: MSFT, Meta: META, Netflix NFLX '''

def fetch_financial_news(api_key, query="NFLX", days_ago=30, language="en", page_size=100):
    """
    Fetch recent financial news articles using the NewsAPI for the last `days_ago` days.

    :param api_key: newsAPI key.
    :param query: Ticker or keyword, e.g., "AAPL" for Apple, NVDA for NVIDIA, QCOM for Qualcomm, NFLX for Netflix
    :param days_ago: How many days back to fetch sentiments for
    :param language: Language code, e.g., 'en' for English.
    :param page_size: Max number of articles per request (up to 100).
    :return: A pandas DataFrame with 'title', 'description', 'publishedAt', etc.
    """
    # Calculate date range: 'days_ago' days from today
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

#Fetch Stock Data (Yahoo Finance)

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # If columns are MultiIndex, drop the extra level
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(1)

    # Compute returns from 'Close'
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data['Target'] = (stock_data['Return'] > 0).astype(int)

    return stock_data

# 3. Methods for sentiment analysis

FIN_POSITIVE_WORDS = {"profit", "growth", "gain", "bullish", "positive", "outperform"}
FIN_NEGATIVE_WORDS = {"loss", "decline", "risk", "bearish", "negative", "underperform"}

def analyze_sentiment_lm(text):
    words = text.lower().split()
    pos_count = sum(word in FIN_POSITIVE_WORDS for word in words)
    neg_count = sum(word in FIN_NEGATIVE_WORDS for word in words)
    return float(pos_count - neg_count)

def analyze_sentiment_finbert(text):
    results = finbert_sentiment(text)
    label = results[0]['label'].lower()  # 'positive', 'negative', or 'neutral'
    score = results[0]['score']         # e.g. 0.993

    if label == 'positive':
        return score
    elif label == 'negative':
        return -score
    else:
        return 0.0

def analyze_sentiment_textblob(text):
    return TextBlob(text).sentiment.polarity


def analyze_sentiment_distilbert(text):
    result = distilbert_sentiment(text)
    score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
    return score

def analyze_sentiment_vader(text):
    return vader_analyzer.polarity_scores(text)['compound']

def analyze_sentiment(text, method='vader'):
    """
    Main dispatcher to analyze text using different sentiment methods.
    Supported methods: 'vader', 'transformers: DistilBERT, FinBERT', 'textblob', 'lm'.
    """
    if method == 'vader':
        return analyze_sentiment_vader(text)
    elif method == 'distilbert':
        return analyze_sentiment_distilbert(text)
    elif method == 'finbert':
        return analyze_sentiment_finbert(text)
    elif method == 'textblob':
        return analyze_sentiment_textblob(text)
    elif method == 'lm':
        return analyze_sentiment_lm(text)
    else:
        raise ValueError("Unsupported sentiment method. Choose from: "
                         "'vader', 'distilbert', 'textblob', 'lm' or finBERT")


# 4. Process Sentiment for different methods

def process_sentiment(df, text_col='Text', method='vader'):
    """
    Add a 'Sentiment' column to df based on the chosen method.
    Assumes 'Text' is a column in df.
    """
    df['Sentiment'] = df[text_col].apply(lambda x: analyze_sentiment(str(x), method))
    return df

def add_technical_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_period=20):
    """
    Compute RSI, MACD, and Bollinger Bands, extra metrics to analyze the stocks and add them as new columns.
    RSI: detect overbought/oversold conditions. RSI > 70 ~ “overbought,” potential for short-term pullback.
    RSI < 30 ~ considered “oversold,” potential for bounce or rally.
    MACD: detect signal changes in momentum and movement of trends.
    Bollinger Bands: reflects volatility (whether or not price is near historical extremes)
    """
    # Ensure 'Close' exists
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' price column to compute indicators.")

    # RSI
    close = df['Close']
    delta = close.diff()

    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ema_up = up.ewm(span=rsi_period, adjust=False).mean()
    ema_down = down.ewm(span=rsi_period, adjust=False).mean()

    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands
    df['BB_middle'] = close.rolling(window=bb_period).mean()
    df['BB_std'] = close.rolling(window=bb_period).std()

    df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
    df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])

    df.drop(columns=['BB_std'], inplace=True)

    return df

# 5. Merge Data

def merge_data(stock_data, sentiment_data):
    # Convert index to a column in stock_data
    stock_data = stock_data.reset_index()

    # Ensure both 'Date' columns are the same type
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.date

    # Merge on 'Date'
    merged_data = pd.merge(stock_data, sentiment_data, on='Date', how='left')

    # Fill any missing sentiment values
    merged_data['Sentiment'].fillna(0, inplace=True)
    return merged_data


# 6. Example Training Models
def train_random_forest(data):
    #features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0)
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0)
    target = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n--- Random Forest ---")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    
    return model, accuracy_score(y_test, y_pred)

def train_logistic_regression(data):
    # features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0)
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0)
    target = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    return model, accuracy_score(y_test, y_pred)

def train_naive_bayes(data):
    # features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0)
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0)
    target = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n--- Naive Bayes ---")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    return model, accuracy_score(y_test, y_pred)

def train_rnn(data):
    # features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0).values
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0)
    target = data['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Reshape for RNN
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
    # features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0).values
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close',
                     'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                     'BB_upper', 'BB_middle', 'BB_lower']].fillna(0)
    target = data['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Reshape for CNN
    X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn  = np.reshape(X_test,  (X_test.shape[0],  X_test.shape[1], 1))

    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
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

# 6. Visualization Helper functionsTools

def plot_stock_and_sentiment(merged_df, stock_label="Stock & Sentiment Over Time"):
    """
    Plots the stock close price and sentiment on two y-axes over time.
    """
    fig, ax1 = plt.subplots(figsize=(10,6))
    # Sort by date if not sorted
    merged_df = merged_df.sort_values(by='Date')

    dates = merged_df['Date']
    close_prices = merged_df['Close']
    sentiment = merged_df['Sentiment']

    # Plot close price on left axis
    ax1.plot(dates, close_prices, color='blue', label='Close Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create second y-axis for sentiment
    ax2 = ax1.twinx()
    ax2.plot(dates, sentiment, color='red', label='Sentiment')
    ax2.set_ylabel('Sentiment', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(stock_label)
    # Optional: combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.show()

def plot_accuracy_scores(acc_dict):
    """
    Plots a bar chart of accuracy scores for different classifiers.
    acc_dict: { 'RandomForest': 0.65, 'NaiveBayes': 0.62, ... }
    """
    classifiers = list(acc_dict.keys())
    scores = list(acc_dict.values())

    plt.figure(figsize=(8,5))
    sns.barplot(x=classifiers, y=scores, palette='viridis')
    plt.ylim(0,1)
    plt.title("Comparison of Classifier Accuracy")
    plt.ylabel("Accuracy")
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
    plt.show()

def plot_sentiment_distribution(df, stock_label="GOOG"):
    """
    Plot a histogram (or KDE) of the Sentiment column.
    """
    plt.figure(figsize=(8,5))
    sns.histplot(df['Sentiment'], kde=True, color='darkgreen')
    plt.title(f"Sentiment Distribution for {stock_label}")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    # Fetch real financial news from NewsAPI
    #api_key = "0341e1c1300a4e62b0fcdd849984821c"   
    api_key = "0e3e57b6a1a34639a40ea9791b64be6f"
    news_df = fetch_financial_news(api_key=api_key, query="GOOG", days_ago=30)
    print("News DataFrame (raw):")
    print(news_df.head())

    
    if not news_df.empty:
        # Combine 'title' + 'description' into a single text field
        news_df['Text'] = news_df['title'].astype(str) + " " + news_df['description'].astype(str)

        # Apply sentiment analysis method from 'vader', 'textblob', 'distilbert', finbert or 'lm')
        news_df = process_sentiment(news_df, text_col='Text', method='finbert')

        # Extract just the date component from 'publishedAt'
        news_df['Date'] = pd.to_datetime(news_df['publishedAt']).dt.date

        # Aggregate daily sentiment
        daily_sent = news_df.groupby('Date')['Sentiment'].mean().reset_index()
    else:
        # If no news is returned, create an empty DataFrame so we can still merge
        daily_sent = pd.DataFrame(columns=['Date', 'Sentiment'])

    # Fetch stock data
    ticker = "NVDA"
    start_date = "2024-12-01"
    end_date = "2025-01-26"
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data = add_technical_indicators(stock_data)

    # Merge the daily sentiment with the stock DataFrame
    merged_data = merge_data(stock_data, daily_sent)

    print("\nMerged DataFrame (Stock + News Sentiment):")
    print(merged_data[['Date','Close','Sentiment']].head(10))

    # Visualize Stock & Sentiment Over Time
    plt.show()
    plot_stock_and_sentiment(merged_data, stock_label=f"{ticker} Stock & Sentiment")

    # Train all the different classifiers & collect accuracy
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
    plt.show()
    plot_accuracy_scores(acc_results)

    # Plot final sentiment distribution
    plt.show()
    plot_sentiment_distribution(merged_data, stock_label=ticker)

    # Extra evaluation metrics listing precicsion, recall, F1 score for all  the different classifiers for GOOG
    rf_model = train_random_forest(merged_data)
    logistic_regression_model = train_logistic_regression(merged_data)
    naive_bayes_model = train_naive_bayes(merged_data)
    rnn_model = train_rnn(merged_data)
    cnn_model = train_cnn(merged_data)

    #Plot all the different stocks together
    tickers = ["AAPL", "NVDA", "AMZN", "QCOM", "GOOG", "MSFT", "META", "NFLX"]
    sentiment_methods = ["vader", "distilbert", "finbert", "textblob", "lm"]
    classifiers = {
        "RandomForest": train_random_forest,
        "LogisticReg": train_logistic_regression,
        "NaiveBayes": train_naive_bayes,
        "RNN": train_rnn,
        "CNN": train_cnn
    }
 
    # We'll store all results in a list of dicts
    results_list = []

    for ticker in tickers:
        print(f"\n=== Processing {ticker} ===")
        # Fetch news
        news_df = fetch_financial_news(api_key, query=ticker, days_ago=30)
        if not news_df.empty:
            news_df['Text'] = news_df['title'].astype(str) + " " + news_df['description'].astype(str)
            news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
        else:
            print(f"No news found for {ticker}, creating empty DataFrame.")
            news_df = pd.DataFrame(columns=['Text','publishedAt'])

        # Fetch stock data
        stock_df = fetch_stock_data(ticker, start_date, end_date)
        stock_df = add_technical_indicators(stock_df)       #additional technical indicators

        for method in sentiment_methods:
            # Process sentiment
            temp_news = news_df.copy()
            if not temp_news.empty:
                temp_news = temp_news.dropna(subset=['Text'])  # drop rows with no text
                temp_news = process_sentiment(temp_news, method=method)
                temp_news['Date'] = temp_news['publishedAt'].dt.date
                daily_sent = temp_news.groupby('Date')['Sentiment'].mean().reset_index()
            else:
                daily_sent = pd.DataFrame(columns=['Date','Sentiment'])

            # Merge
            merged_df = merge_data(stock_df, daily_sent)

            # Plot the stock & sentiment for this method/ticker
            plt.show()
            plot_stock_and_sentiment(merged_df, stock_label=f"{ticker}-{method}")

            # Train classifiers
            for clf_name, clf_func in classifiers.items():
                model, acc = clf_func(merged_df)
                print(f"{ticker} | {method} | {clf_name} => Accuracy: {acc:.3f}")

                results_list.append({
                    "Ticker": ticker,
                    "Method": method,
                    "Classifier": clf_name,
                    "Accuracy": acc
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    print("\n--- Final Results DataFrame ---")
    print(results_df.head(100))

    # Plot of accuracy by (Method, Classifier)
    plt.figure(figsize=(12,6))
    sns.barplot(data=results_df, x="Method", y="Accuracy", hue="Classifier", errorbar=None)
    plt.ylim(0,1)
    plt.title("Accuracy Across Sentiment Methods & Classifiers (All Stocks Aggregated)")
    plt.show()

    # break it down by ticker
    g = sns.catplot(data=results_df, x="Method", y="Accuracy", hue="Classifier",
                    col="Ticker", col_wrap=4, kind="bar", sharey=False, errorbar=None)
    g.set_titles("{col_name}")
    g.set(ylim=(0,1))
    g.fig.suptitle("Accuracy by Ticker, Method, and Classifier")
    g.fig.subplots_adjust(top=0.88)
    plt.show()




   

