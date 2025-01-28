# within-takehome

For artificially created sentiments, run: python sentiment_dummy.py

For real world financial news sentiments, run: python sentiment.py


Approach: 
We use sentiment analyzers like vader, textblob, loughran-mcdonald, distilbert, finbert to analyze real
world sentiments from NewsAPI, we merge those sentiments with corresponding stock data from Yahoo finance.

We then use classifiers such as Random Forest, Naive Bayes, Logistic Regresssion, Convolulional Neural networks
 and Recurrent Neural Networks to predict stock movement and closing stock price using technical indicators like
 Relative Strength Index(RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands in addition to daily 
 high, low, volume and closing prices