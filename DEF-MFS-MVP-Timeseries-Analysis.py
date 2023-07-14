# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:22:55 2023

@author: aakas
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import plotly.io as pio

import os
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.dates as dates
import plotly.express as px
from plotly.subplots import make_subplots

class TimeSeriesAnalysis_Telsa:
    def __init__(self):
        self.df = pd.read_excel("C:/Users/aakas/Documents/Co-op/Week 5/TSLA.xlsx")
        self.df = self.df.drop("Unnamed: 0",axis=1)
        self.train_data, self.test_data = self.df[0:int(len(self.df)*0.7)], self.df[int(len(self.df)*0.7):]
        self.training_data = self.train_data['Close'].values
        self.test_data = self.test_data['Close'].values
        
        
        self.history = [x for x in self.training_data]
        self.model_predictions = []
        #self.model = ARIMA(self.history, order=(4,1,0))
        
    def AutoCorrelationPlot(self,lag):
        plt.figure()
        lag_plot(self.df['Open'], lag=lag)
        plt.title('TESLA Stock - Autocorrelation plot with lag = 3')
        plt.show()
        
    def StockPriceOverTime(self):
        plt.figure(figsize=(20,12))
        plt.plot(self.df["Date"], self.df["Close"])
        plt.title("TESLA stock price over time")
        plt.xlabel("time")
        plt.ylabel("price")
        plt.show()
        
    def model(self):
        
        N_test_observations = len(self.test_data)
        
        
        for time_point in range(N_test_observations):
            model = ARIMA(self.history, order=(4,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            self.model_predictions.append(yhat)
            true_test_value = self.test_data[time_point]
            self.history.append(true_test_value)
            
            
        MSE_error = mean_squared_error(self.test_data, self.model_predictions)
        print('Testing Mean Squared Error is {}'.format(MSE_error))
        
        test_set_range = self.df[int(len(self.df)*0.7):].index
        
        plt.figure(figsize=(20,12))

        plt.plot(test_set_range, self.model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
        plt.plot(test_set_range, self.test_data, color='red', label='Actual Price')
        
        plt.title('TESLA Prices Prediction')
        plt.xlabel('Date')
        plt.ylabel('Prices')
        plt.legend()
        plt.show()
        
    def get_prediction_dashboard(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df["Date"], y=self.df["Close"], name="Actual Price"))
        fig.add_trace(go.Scatter(x=self.df["Date"][len(self.train_data):], y=self.model_predictions, name="Predicted Price"))
        fig.update_layout(
            title="TESLA stock price: Actual vs Predicted",
            xaxis_title="Time",
            yaxis_title="Price",
            legend=dict(x=0, y=1, traceorder="normal"),
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"visible": [True, True]}],
                            label="All",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [True, False]}],
                            label="Actual Price",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [False, True]}],
                            label="Predicted Price",
                            method="update"
                        )
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
        pio.renderers.default = "browser"
        fig.show()
        
        
class TimeSeriesAnalysis_Ford:
    def __init__(self):
        self.df = pd.read_csv("C:/Users/aakas/Documents/Co-op/Week 5/F.csv")
        
    def results(self):
        TimeSeriesAnalysis_Ford.plot_data(self)
        x_train,y_train,train_data,test_data,prediction_days,scaler=TimeSeriesAnalysis_Ford.train_test_split(self)
        model = TimeSeriesAnalysis_Ford.LSTM_model(self, x_train, y_train, train_data, test_data, prediction_days, scaler)
        TimeSeriesAnalysis_Ford.predict_test(self,model,train_data,test_data,prediction_days,scaler)
        
    def plot_data(self):
        fig=make_subplots(specs=[[{"secondary_y":False}]])
        fig.add_trace(go.Scatter(x=self.df['Date'],y=self.df['Open'].rolling(window=7).mean(),name="Ford"),secondary_y=False,)
        fig.update_layout(autosize=False,width=900,height=500,title_text="Ford")
        fig.update_xaxes(title_text="year")
        fig.update_yaxes(title_text="prices",secondary_y=False)
        fig.show()
        
    def train_test_split(self):
        n=len(self.df)
        train_data=self.df[(n//20)*15:(n//20)*19]
        test_data=self.df[(n//20)*19:]
        
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(train_data['Open'].values.reshape(-1,1))
        
        prediction_days = 30

        x_train = []
        y_train = []
        
        for x in range(prediction_days, len(scaled_data)-20):      ######
            x_train.append(scaled_data[x-prediction_days:x, 0])
            y_train.append(scaled_data[x+20, 0])      ###### predict 20 days after
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train,y_train,train_data,test_data,prediction_days,scaler
        
    def LSTM_model(self,x_train,y_train,train_data,test_data,prediction_days,scaler):
        
        model = Sequential()    
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])
        
        checkpointer = ModelCheckpoint(filepath = 'weights_best.hdf5', verbose = 1, save_best_only = True)
        his=model.fit(x_train,y_train,epochs=30,batch_size=32,callbacks=[checkpointer])
        
        
        plt.plot(his.history['loss'])
        plt.plot(his.history['accuracy'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss','accuracy'], loc='upper right')
        plt.show()
        
        return model
        
    def predict_test(self,model,train_data,test_data,prediction_days,scaler):
            actual_prices = test_data['Open'].values
            total_dataset = pd.concat((train_data['Open'], test_data['Open']), axis=0)
            
            model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
            model_inputs = model_inputs.reshape(-1,1)
            model_inputs = scaler.transform(model_inputs)
            
            
            x_test = []
            for x in range(prediction_days,len(model_inputs)):
                x_test.append(model_inputs[x-prediction_days:x,0])
            
            x_test = np.array(x_test)
            x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
            
            predicted_prices = model.predict(x_test)
            predicted_prices = scaler.inverse_transform(predicted_prices)
            
            plt.plot(actual_prices, color='black', label=f"Actual price")
            plt.plot(predicted_prices, color= 'green', label=f"Predicted 20-days-after price")
            plt.title(f"Ford Stock")
            plt.xlabel("Days in test period")
            plt.ylabel(f"Price")
            plt.legend()
            plt.show()
            
            test_data['predict']=predicted_prices
            
            fig=make_subplots(specs=[[{"secondary_y":False}]])
            fig.add_trace(go.Scatter(x=train_data['Date'],y=train_data['Open'],name="Train Actual"),secondary_y=False,)
            fig.add_trace(go.Scatter(x=test_data['Date'],y=test_data['Open'],name="Test Actual"),secondary_y=False,)
            fig.add_trace(go.Scatter(x=test_data['Date'],y=test_data['predict'],name="Predicted 20-days after price"),secondary_y=False,)
            fig.update_layout(autosize=False,width=900,height=500,title_text="Ford")
            fig.update_xaxes(title_text="year")
            fig.update_yaxes(title_text="prices",secondary_y=False)
            fig.show()