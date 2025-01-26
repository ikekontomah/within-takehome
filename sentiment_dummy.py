import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from newsapi import NewsApiClient

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

# import nltk
# # from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from textblob import TextBlob
import datetime as dt

vader_analyzer = SentimentIntensityAnalyzer()

# Initialize transformers sentiment analysis pipeline (optional)
transformer_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased")


def fetch_financial_news(api_key, query="AAPL", days_ago=30, language="en", page_size=100):
    """
    Fetch recent financial news articles using the NewsAPI for the last `days_ago` days.

    :param api_key: Your NewsAPI key.
    :param query: Ticker or keyword, e.g., "AAPL" for Apple.
    :param days_ago: How many days back to fetch (default: 30).
    :param language: Language code, e.g., 'en' for English.
    :param page_size: Max number of articles per request (up to 100).
    :return: A pandas DataFrame with the fetched articles.
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


### 1. Collect Financial Data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # If columns are MultiIndex, drop the extra level
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(1)

    # Compute returns from 'Close'
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data['Target'] = (stock_data['Return'] > 0).astype(int)

    return stock_data

############################################
# 2. Multiple Sentiment Methods
############################################

# 2.1 Loughran-McDonald-like Dictionary (Simplified Demo)
# In practice, you'd load the full dictionary from a file.
FIN_POSITIVE_WORDS = {"profit", "growth", "gain", "bullish", "positive", "outperform"}
FIN_NEGATIVE_WORDS = {"loss", "decline", "risk", "bearish", "negative", "underperform"}

def analyze_sentiment_lm(text):
    """
    A simplified example of analyzing sentiment using a
    Loughran-McDonald-style dictionary approach.

    Returns a 'compound-like' score by subtracting negative counts from positive counts.
    """
    words = text.lower().split()
    pos_count = sum(word in FIN_POSITIVE_WORDS for word in words)
    neg_count = sum(word in FIN_NEGATIVE_WORDS for word in words)
    return float(pos_count - neg_count)  # simple difference

def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob.
    Returns a polarity score in the range [-1.0, 1.0].
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


# Example Loughran-McDonald-like Dictionary
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
    """
    Main dispatcher to analyze text using different sentiment methods.
    Supported methods: 'vader', 'transformer', 'textblob', 'lm'.
    """
    if method == 'vader':
        score = analyze_sentiment_vader(text)
    elif method == 'transformer':
        score = analyze_sentiment_transformers(text)
    elif method == 'textblob':
        score = analyze_sentiment_textblob(text)
    elif method == 'lm':
        score = analyze_sentiment_lm(text)
    else:
        raise ValueError("Unsupported sentiment method. Choose from: "
                         "'vader', 'transformer', 'textblob', or 'lm'.")
    return score


# ### 2. Analyze Sentiment
# def analyze_sentiment(text, method='vader'):
#     """Analyze sentiment of a text using VADER or Transformers."""
#     if method == 'vader':
#         score = vader_analyzer.polarity_scores(text)['compound']
#     elif method == 'transformer':
#         result = transformer_sentiment(text)
#         score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
#     return score

def process_sentiment(data, method='transformer'):
    """Add sentiment scores to a DataFrame."""
    data['Sentiment'] = data['Text'].apply(lambda x: analyze_sentiment(x, method))
    return data


def merge_data(stock_data, sentiment_data):
    # Convert index to a column in stock_data
    stock_data = stock_data.reset_index()

    # Ensure both 'Date' columns are the same type
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])

    # Merge on 'Date'
    merged_data = pd.merge(stock_data, sentiment_data, on='Date', how='left')

    # Fill any missing sentiment values
    merged_data['Sentiment'].fillna(0, inplace=True)
    return merged_data


### 4. Train Predictive Model
def train_model(data):
    """Train a machine learning model to predict stock movements."""
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0)
    target = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    return model

############################################
# 4. Different Model Training Functions
############################################

def train_random_forest(data):
    """
    Train a RandomForestClassifier on the merged data.
    """
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0)
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
    
    return model

def train_logistic_regression(data):
    """
    Train a LogisticRegression model on the merged data.
    """
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0)
    target = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    return model

def train_naive_bayes(data):
    """
    Train a Naive Bayes classifier (GaussianNB) on the merged data.
    (Works fine for numeric features, though more common with text features.)
    """
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0)
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

    return model


############################################
# 5. Simple RNN (LSTM) and CNN Demonstration
############################################

def train_rnn(data):
    """
    Train a very simple RNN (LSTM) on the merged data.
    For demonstration, each day is treated as a single time step (shape = [batch, 1, features]).
    
    In a real scenario, you'd create sequences of multiple days for each sample.
    """
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0).values
    target = data['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Reshape from [samples, features] to [samples, timesteps=1, features]
    X_train_rnn = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn  = np.reshape(X_test,  (X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(16, input_shape=(1, X_train.shape[1])))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluation
    loss, accuracy = model.evaluate(X_test_rnn, y_test, verbose=0)
    y_pred = (model.predict(X_test_rnn) > 0.5).astype(int).flatten()

    print("\n--- LSTM RNN ---")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score: {accuracy:.3f}")

    return model

def train_cnn(data):
    """
    Train a simple 1D CNN on the merged data.
    Similar note: each sample has shape [1, features], so this is not truly capturing
    time-series aspects. For a real CNN approach, you'd typically use multi-day windows.
    """
    features = data[['Sentiment', 'Volume', 'Open', 'High', 'Low', 'Close']].fillna(0).values
    target = data['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Reshape from [samples, features] to [samples, timesteps=1, features] 
    # to apply a 1D convolution.
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

    # Evaluation
    loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
    y_pred = (model.predict(X_test_cnn) > 0.5).astype(int).flatten()

    print("\n--- 1D CNN ---")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score: {accuracy:.3f}")

    return model

### 5. Example Pipeline Execution
if __name__ == "__main__":
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # # Example (fake) sentiment_data
    # sentiment_data = pd.DataFrame({
    #     'Date': pd.date_range(start='2023-01-01', periods=10),
    #     'Text': ['Good news about Apple'] * 5 + ['Bad news about Apple'] * 5,
    #     'Sentiment': [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]  # sample numeric sentiments
    # })

    sentiment_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'Text': [
            "Apple releases groundbreaking product",
            "Positive earnings report from Apple",
            "Apple is facing supply chain issues",
            "Mixed analyst opinions on Apple",
            "Apple invests in new technology",
            "Concerns about Apple's overseas production",
            "Apple gains market share",
            "Apple facing lawsuit",
            "Strong iPhone sales report",
            "Rumors about upcoming Apple event"
        ]
    })

    sentiment_data = process_sentiment(sentiment_data, 'lm')
    
    # 4) Merge stock & sentiment data
    merged_data = merge_data(stock_data, sentiment_data)
    # print("\nMerged Data Preview:")
    # print(merged_data.head(10))

 
    # random_forest_model = train_random_forest(merged_data)
    #logistic_regression_model = train_logistic_regression(merged_data)
    # nb_model = train_naive_bayes(merged_data)
    # rnn_model = train_rnn(merged_data)
    cnn_model = train_cnn(merged_data)


