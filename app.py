from flask import Flask, render_template, request
import yfinance as yf
import os
import pandas as pd
import numpy as np
import pandas_datareader as web
import matplotlib.pyplot as plt

import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from datetime import date, datetime, timedelta
from keras.models import load_model
import math

app = Flask(__name__)

def printGraph(stock_name, pred):
  df = yf.download(stock_name, date.today() - timedelta(15))
  df.drop(['High','Low','Open','Volume','Adj Close'],axis=1,inplace=True)
  df['Date'] = df.index
  dff = df.tail(1)
  check = str(dff.tail(1).index)[16:26] == str(date.today())
  if check:
    date_dff = pd.date_range(start=dt.date.today() + timedelta(1), periods = 3, freq='D')
  else:
    date_dff = pd.date_range(start=dt.date.today(), periods = 3, freq='D')
  new_dff = pd.DataFrame(data=date_dff,columns=['Date'])
  new_dff.index = new_dff.Date
  new_dff['Close'] = pred
  dff = dff.append(new_dff.tail(3))
  #print(dff)
  dff.drop(columns=['Date'],axis=1,inplace=True)
  plt.figure(figsize=(20,8))
  plt.grid()
  plt.title('CLOSE PRICE')
  #plt.plot(train_data["Close"],color='blue',label='Close')
  plt.plot(df['Close'],label='Actual Close',color='blue',marker = '*')
 # plt.plot(dff['Close'],'g--',label='Predicted Close',marker = '+')
  plt.plot(dff['Close'],'green',label='future Predicion',marker = 'o')
  plt.legend()
  plt.savefig(fname='static/image.jpg',bbox_inches='tight')
  #print(df)
  
def predict(stock_name, isIssue = False):
  df = yf.download(stock_name, '2022-01-01')
  df.drop(['High','Low','Open','Volume','Adj Close'],axis=1,inplace=True)
  scaler = MinMaxScaler(feature_range=(0,1))
  # print(df.tail())
  inputs_data = df.values
  inputs_data = inputs_data.reshape(-1,1)
  inputs_data = scaler.fit_transform(inputs_data)
  inputs_data = inputs_data.reshape(-1)
  if isIssue == False:
    inputs_data = inputs_data[-14:]
  else:
    inputs_data = inputs_data[-15:-1]
  future_data = list(inputs_data)   
  # print(inputs_data)
  future_prediction = []
  model = load_model('model.h5')
  for i in range(3):
    future_dataa = np.array(future_data,dtype='float32')
    future_data_reshaped = future_dataa.reshape(1,14,1)
    predicted_price = model.predict(future_data_reshaped)
    # print(predicted_price)
    future_data.pop(0)
    future_data.append(predicted_price)
    predicted_price = scaler.inverse_transform(predicted_price)
    predicted_price = predicted_price.reshape(1)
    predicted_price = float(predicted_price)
    future_prediction.append(predicted_price)
    if math.isnan(predicted_price):
      predict(stock_name, True)
      return
  printGraph(stock_name, future_prediction)
  return future_prediction

@app.route('/showChart')
def work():
  return render_template('showChart.html')

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data = request.form['stock_name']
    future = predict(data)
    return render_template('after.html', data1 = future[0], data2 = future[1], data3 = future[2])
    
if __name__ == "__main__":
    app.run(debug=False)
    
    
    
