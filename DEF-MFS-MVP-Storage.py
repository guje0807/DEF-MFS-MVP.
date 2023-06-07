# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:24:10 2023

@author: aakas
"""

import boto3
import json

class dataStorage:
    def __int__(self):
        print("Instantiating Data Storage Object")
        self.aws_access_key_id = ""
        self.aws_secret_access_key=""
        self.bucket_name = ""
        
    def read_config(self):
        print("Reading config file")
        with open("C:/Users/aakas/Documents/Co-op/Week 2/DEF-MFS-MVP-Configuration.json") as f:
            data = json.load(f)
            
        self.aws_access_key_id=data['aws_access_key_id']
        self.aws_secret_access_key = data['aws_secret_access_key']
        self.bucket_name = data['bucket_name']
        
    
    def upload_object(self,key,ticker_symbol):
        try:
            print("Creating the S3 Client")
            s3 = boto3.client("s3",
                     aws_access_key_id=self.aws_access_key_id,
             aws_secret_access_key= self.aws_secret_access_key)
            
            print("Uploading the Object to S3")
        
        
            s3.upload_file(
                Filename = key,
                Bucket="stock-date",
                Key=ticker_symbol,
                )
            return True
        
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            return False
        
        
    
        