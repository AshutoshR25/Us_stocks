from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import os
import plotly.graph_objects as go
import matplotlib
import tensorflow
from tensorflow import keras
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
# from wordcloud import WordCloud, STOPWORDS
import yfinance as yf
from datetime import date
from sklearn.preprocessing import MinMaxScaler
nltk.download('vader_lexicon') #required for Sentiment Analysis

import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = "123"

DATA_PATH = os.path.join(os.path.dirname(__file__), "static", "data")
IMAGES_PATH = os.path.join(os.path.dirname(__file__), "static", "images")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "static", "models")


users = [
    {"username": "user1", "password": "user1"},
    {"username": "user2", "password": "user2"},
    {"username": "user3", "password": "user3"},
    {"username": "user4", "password": "user4"},
    {"username": "user5", "password": "user5"},
]


@app.route("/", methods=['GET', 'POST'])
def landingpage():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # if(username == "admin" and password == "admin123"):
        for user in users:
            if (username == user['username'] and password == user['password']):
                session['username'] = username
                return render_template("index.html", username=session['username'])
        else:
            return render_template("login.html", msg="Invalid Credentials!")
    else:
        return render_template("login.html")

@app.route("/logout", methods=['GET', 'POST'])
def logout():
    session.pop('username', None)
    return render_template("login.html")


@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'POST' and session['username'] != None:
        # getting selected stock name
        stock = request.form['stockname']
        

        get_stock_prediction(stock)
        sentiment_df , recommendation_text, news_array = full_detail(stock)
    
        return render_template("index.html", stock=stock, imageName = str(stock+'.png'), username=session['username'], sentiment_df=sentiment_df.to_html(index=False).replace('<th>', '<th style="text-align:center">'), recommendation_text=recommendation_text, news_array=news_array)
    else:
        return render_template("index.html", username=session['username'])


def get_stock_prediction(stock_s):

  today = date.today()
  if stock_s == "Dominos":
      stock_s = "DPZ"
  data = yf.download(stock_s,'2020-01-01',today)
  # Plot the close prices
  df = data.copy()
  #Set Target Variable
  output_var = pd.DataFrame(df['Adj Close'])
  #Selecting the Features
  features = ['Open', 'High', 'Low', 'Volume']
  #Scaling
  scaler = MinMaxScaler()
  feature_transform = scaler.fit_transform(df[features])
  feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
  feature_transform.head()

  #Splitting to Training set and Test set
  timesplit= TimeSeriesSplit(n_splits=10)
  for train_index, test_index in timesplit.split(feature_transform):
          X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
          y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()
  #Process the data for LSTM
  trainX =np.array(X_train)
  testX =np.array(X_test)
  X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
  X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

  # load the model
  loaded_model =  keras.models.load_model(MODELS_PATH+'\\'+stock_s+'_model.h5')

  #LSTM Prediction
  y_pred= loaded_model.predict(X_test)

  #Predicted vs True Adj Close Value – LSTM
  plt.clf()
  plt.plot(y_test, label='True Value')
  plt.plot(y_pred, label='LSTM Value')
  plt.title(f"Prediction for {stock_s}")
  plt.xlabel('Time Scale')
  plt.ylabel('Scaled USD')
  plt.legend()
  plt.savefig(IMAGES_PATH+'\\'+'prediction.png')
  plt.clf()


# Sentiment Analysis
def percentage(part, whole):
    return 100 * float(part)/float(whole)

news_df = pd.DataFrame()
def full_detail(company_name):
    global news_df
    now = dt.date.today()
    now = now.strftime('%m-%d-%Y')
    yesterday = dt.date.today() - dt.timedelta(days = 1)
    yesterday = yesterday.strftime('%m-%d-%Y')

    nltk.download('punkt')
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10
    if company_name != '':
        print(
            f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')

        # Extract News with Google News
        googlenews = GoogleNews(start=yesterday, end=now)
        googlenews.search(company_name)
        result = googlenews.result()
        # store the results
        df = pd.DataFrame(result)
        news_column = df['title'].values
        # limit to maximum number of rows in dataframe
        max_rows = min(5, len(news_column))
        news_array = news_column[:max_rows]

    try:
        list = []  # creating an empty list
        for i in df.index:
            dict = {}  # creating an empty dictionary to append an article in every single iteration
            # providing the link
            article = Article(df['link'][i], config=config)
            try:
                article.download()  # downloading the article
                article.parse()  # parsing the article
                article.nlp()  # performing natural language processing (nlp)
            except:
                pass
            # storing results in our empty dictionary
            dict['Date'] = df['date'][i]
            dict['Media'] = df['media'][i]
            dict['Title'] = article.title
            dict['Article'] = article.text
            dict['Summary'] = article.summary
            dict['Key_words'] = article.keywords
            list.append(dict)
        check_empty = not any(list)
        # print(check_empty)
        if check_empty == False:
            news_df = pd.DataFrame(list)  # creating dataframe
            # print(news_df)

    except Exception as e:
        # exception handling
        print("exception occurred:" + str(e))
        print('Looks like, there is some error in retrieving the data, Please try again or try with a different ticker.')



# Assigning Initial Values
    positive = 0
    negative = 0
    neutral = 0
    # Creating empty lists
    news_list = []
    neutral_list = []
    negative_list = []
    positive_list = []

    # Iterating over the tweets in the dataframe
    for news in news_df['Summary']:
        news_list.append(news)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
        neg = analyzer['neg']
        neu = analyzer['neu']
        pos = analyzer['pos']
        comp = analyzer['compound']

        if neg > pos:
            # appending the news that satisfies this condition
            negative_list.append(news)
            negative += 1  # increasing the count by 1
        elif pos > neg:
            # appending the news that satisfies this condition
            positive_list.append(news)
            positive += 1  # increasing the count by 1
        elif pos == neg:
            # appending the news that satisfies this condition
            neutral_list.append(news)
            neutral += 1  # increasing the count by 1

    # percentage is the function defined above
    positive = percentage(positive, len(news_df))
    negative = percentage(negative, len(news_df))
    neutral = percentage(neutral, len(news_df))

    # Converting lists to pandas dataframe
    news_list = pd.DataFrame(news_list)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)
    # using len(length) function for counting
    sentiment_dict = {
    'Positive Sentiment': [len(positive_list)],
    'Neutral Sentiment': [len(neutral_list)],
    'Negative Sentiment': [len(negative_list)]
    }

    sentiment_df = pd.DataFrame(sentiment_dict)

    # Creating PieCart
    labels = ['Positive ['+str(round(positive))+'%]', 'Neutral ['+str(
        round(neutral))+'%]', 'Negative ['+str(round(negative))+'%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'blue', 'red']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.style.use('default')
    plt.legend(labels)
    plt.title("Sentiment Analysis Result for stock= "+company_name+"")
    plt.axis('equal')
    plt.savefig(IMAGES_PATH+'\\'+'plot.png')
    recommendation_text = ""

    if positive > negative:
        recommendation_text = f"It has more probability to store in the stock {company_name} with percentage {round(positive,2)} %"
    elif negative < positive:
       recommendation_text = f"It has less probability to store in the stock {company_name} with percentage {round(negative,2)} %"
    else:
        recommendation_text = f"It has equal probability to store in the stock {company_name}"

    return sentiment_df , recommendation_text, news_array



if __name__ == '__main__':
    app.run(debug=True)
