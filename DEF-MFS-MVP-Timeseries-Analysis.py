# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:22:55 2023

@author: aakas
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class TimeSeriesAnalysis_Telsa:
    def __init__(self):
        self.df = pd.read_excel("C:/Users/aakas/Documents/Co-op/Week 5/TSLA.xlsx")
        self.df = self.df.drop("Unnamed: 0",axis=1)
        
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
        train_data, test_data = self.df[0:int(len(self.df)*0.7)],self.df[int(len(self.df)*0.7):]
        training_data = train_data['Close'].values
        test_data = test_data['Close'].values
        
        history = [x for x in training_data]
        model_predictions = []
        N_test_observations = len(test_data)
        
        
        for time_point in range(N_test_observations):
            model = ARIMA(history, order=(4,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            model_predictions.append(yhat)
            true_test_value = test_data[time_point]
            history.append(true_test_value)
            
            
        MSE_error = mean_squared_error(test_data, model_predictions)
        print('Testing Mean Squared Error is {}'.format(MSE_error))
        
        test_set_range = self.df[int(len(self.df)*0.7):].index
        
        plt.figure(figsize=(20,12))

        plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
        plt.plot(test_set_range, test_data, color='red', label='Actual Price')
        
        plt.title('TESLA Prices Prediction')
        plt.xlabel('Date')
        plt.ylabel('Prices')
        plt.legend()
        plt.show()