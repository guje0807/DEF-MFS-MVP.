# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 08:23:00 2023

@author: aakas
"""



from prophet import Prophet

import pandas as pd

class Timeseries_Forecasting:
    def __init__(self):
        self.df = pd.read_csv("C:/Users/aakas/Documents/Co-op/Week 5/F.csv")
        
    def implement_model(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])  # Convert the 'Date' column to datetime
        self.df = self.df[['Date', 'Close']]
        self.df = self.df.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        model = Prophet(daily_seasonality=True)  # Enable daily seasonality
        model.fit(self.df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=12, freq='M')  # Generate monthly date range for 1 year
        forecast = model.predict(future)
        
        # Plot the forecast
        model.plot(forecast, xlabel='Date', ylabel='Price')