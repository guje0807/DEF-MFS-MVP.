# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:02:54 2023

@author: aakas
"""

from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
from dash import dcc
import pandas as pd

class interactive_viz:
    def generate_dashboard(self,df_1,df_2):
        print("Inside Generate_dashboard")
        app = Dash(__name__)

        # Define the layout of the app
        app.layout = html.Div([
            html.H1("Stock Price Comparison"),
            html.Label("Select Stock:"),
            dcc.Dropdown(
                id='stock-dropdown',
                options=[
                    {'label': 'Tesla', 'value': 'tesla'},
                    {'label': 'Ford', 'value': 'ford'}
                ],
                value='tesla'
            ),
            dcc.Graph(id='stock-chart')
        ])
        
        # Define the callback function to update the chart based on the selected stock
        @app.callback(
            Output('stock-chart', 'figure'),
            Input('stock-dropdown', 'value')
        )
        def update_chart(selected_stock):
            print("Inside Update Chart")
            if selected_stock == 'tesla':
                df = df_2
                title = 'Tesla Stock Price (2021-2022)'
            elif selected_stock == 'ford':
                df = df_1
                title = 'Ford Stock Price (2021-2022)'
            else:
                df = pd.DataFrame(columns=['Date'])
                title = 'No stock selected'
        
            fig = go.Figure()
            if not df.empty:
                fig.add_trace(go.Candlestick(x=df['Date'],open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'],
                                             increasing_line_color='green',
                                             decreasing_line_color='red'))
        
            fig.update_layout(title=title,
                              xaxis_title="Date",
                              yaxis_title="Price",
                              template="plotly_white")
        
            return fig
        
        app.run_server()
