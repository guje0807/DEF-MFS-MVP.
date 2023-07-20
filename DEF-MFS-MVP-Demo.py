# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:05:39 2023

@author: aakas


"""


import pandas as pd
from prophet import Prophet
import plotly.io as pio
import plotly.graph_objects as go

class Timeseries_Demo:
    def __init__(self):
        self.df = pd.read_csv("C:/Users/aakas/Documents/Co-op/Week 5/F.csv")
        
    def implement_demo(self):
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
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df["Date"], y=self.df["Close"], name="Actual Price"))
        fig.add_trace(go.Scatter(x=self.df["Date"][len(future):], y=forecast, name="Predicted Price"))
        fig.update_layout(
            title="Telsa stock price: Actual vs Predicted",
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