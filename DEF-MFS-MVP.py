# -*- coding: utf-8 -*-
"""
Created on Wed May 31 08:19:16 2023

@author: aakas
"""

import pandas as pd
import argparse
import yfinance as yf


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
    

#Start

print("Usage: DEF-MFS-MVP.py -t ticker_sysmbol -s start_date -e end_date") #Printing the Usage of the file 

argParser = argparse.ArgumentParser() 
argParser.add_argument("-t", "--ticker_symbol", help="Ticker Symbol") # Adding Ticker Symbol Argument 
argParser.add_argument("-s", "--start_date", help="Start Date") # Adding Start Date Argument
argParser.add_argument("-e", "--end_date", help="End Date") # Adding End Date Argument

args = argParser.parse_args() # Loading the Arguments to a variable

print("Passed Argumnets are:", args) # Printing the Arguments 

stock_data = stock(args.ticker_symbol, args.start_date, args.end_date) #Declaring the Object with ticker symbol,start date and end date.

df = stock_data.download_stock_info() # Calling the Class member function

df.to_excel(f"{stock_data.ticker_symbol}.xlsx") #Downloading the stock data to excel file

print(df.head(10)) # Printing the top 10 rows of the stock data. 

#End
    
    
