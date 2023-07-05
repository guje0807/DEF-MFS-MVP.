# -*- coding: utf-8 -*-
"""
Created on Wed May 31 08:19:16 2023

@author: aakas
"""
import pandas as pd
import argparse
import yfinance as yf
import matplotlib.pyplot as plt
import glob


a = __import__("DEF-MFS-MVP-Storage")
b = __import__("DEF-MFS-MVP-StatisticalAnalysis")
c = __import__("DEF-MFS-MVP-Visualization")
d = __import__("DEF-MFS-MVP-InteractiveVisualization")
e = __import__("DEF-MFS-MVP-Timeseries-Analysis")

#Class Definition for Specific Stock company
class stock:
    def __init__(self, ticker_symbol, start_date, end_date):   #Constructor 
        print("Instantiating Object")
        self.ticker_symbol = ticker_symbol #Ticker Symbol Variable
        self.start_date = start_date # Start Date Variable
        self.end_date = end_date # End Date Variable 
        self.stock_data = None
        
    #Method to download the stock data using y finance module.    
    def download_stock_info(self):
        print("Downloading Stock Data for {}".format(self.ticker_symbol)) 
        self.stock_data = yf.download(self.ticker_symbol, start=self.start_date, end=self.end_date)
        df = pd.DataFrame(self.stock_data)
        df.reset_index(inplace=True)
        print("Returning Stock Data")
        return df


def download_data():
    argParser = argparse.ArgumentParser() 
    argParser.add_argument("-t", "--ticker_symbol", help="Ticker Symbol") # Adding Ticker Symbol Argument 
    argParser.add_argument("-s", "--start_date", help="Start Date") # Adding Start Date Argument
    argParser.add_argument("-e", "--end_date", help="End Date") # Adding End Date Argument
    
    args = argParser.parse_args() # Loading the Arguments to a variable
    
    print("Passed Argumnets are:", args) # Printing the Arguments 
    
    stock_data = stock(args.ticker_symbol, args.start_date, args.end_date) #Declaring the Object with ticker symbol,start date and end date.
    
    df = stock_data.download_stock_info() # Calling the Class member function
    
    fileName = f"{stock_data.ticker_symbol}.xlsx"
    
    df.to_excel(f"{stock_data.ticker_symbol}.xlsx") #Downloading the stock data to excel file
    
    #print(df.head(10)) # Printing the top 10 rows of the stock data.


#Function to store data to S3.
def store_data(fileName,ticker_symbol):
    ds = a.dataStorage()
    ds.read_config()
    
    if(ds.upload_object(fileName,ticker_symbol)):
        print("File Upload Sucessfull!!")
    else:
        print("File Upload Failed!!")

#Function to get stats
def get_statistics():
    df = pd.read_excel("C:/Users/aakas/Documents/Co-op/Week 5/Ford.xlsx")
    cols = df.columns
    l = dict.fromkeys(cols[2:])
    for i in cols[2:]:
        s = b.stats()
        l[i] = s.cal_stats(df[i])
    
    stats_df = pd.DataFrame.from_dict(l,orient='index', columns=["Min","Max","Range","Mean","Median","Std","Var"])
    print(stats_df)

#Function to get Plots        
def get_viz():
    df_ford = pd.read_excel("C:/Users/aakas/Documents/Co-op/Week 5/Ford.xlsx")
    df_tsla = pd.read_excel("C:/Users/aakas/Documents/Co-op/Week 5/TSLA.xlsx")
    
    cols = df_ford.columns
    date = df_ford['Date']
    
    for i in cols[2:]:
        v = c.viz()
        
        print("Plot for the columns {}".format(i))
        a = df_ford[i]
        b = df_tsla[i]
        
        v.line_chart(a,b,date)

def interactive_visualizations():
    print("Inside Interactive viz")
    df_ford = pd.read_excel("C:/Users/aakas/Documents/Co-op/Week 5/Ford.xlsx")
    df_tsla = pd.read_excel("C:/Users/aakas/Documents/Co-op/Week 5/TSLA.xlsx")
        
    v = d.interactive_viz()
    v.generate_dashboard(df_ford,df_tsla)

def modelling():
    print("Inside Modelling")
    
    m = e.TimeSeriesAnalysis_Telsa()
    m.AutoCorrelationPlot(lag = 3)
    m.StockPriceOverTime()
    m.model()

       
#Main Function
def main():
    #print("Usage: DEF-MFS-MVP.py -t ticker_sysmbol -s start_date -e end_date") #Printing the Usage of the file 
    
    #download Data
    #download_data()
    
    # Uploading data to S3
    #store_data(fileName,stock_data.ticker_symbol)
    
    # Function to get statistical values
    #get_statistics()
    
    #Function to get Plots
    #get_viz()
    
    #Function to get run Plotly dashboard
    #interactive_visualizations()
    
    #Function to implement ARIMA model on Tesla Stock
    modelling()

#Start

main()

#End
    
    
