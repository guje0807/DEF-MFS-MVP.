# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:35:37 2023

@author: aakas
"""
import pandas as pd

class stats:
    def __init__(self):
        self.min = 0
        self.max = 0
        self.range = 0
        self.mean = 0
        self.median = 0
        self.std = 0
        self.variance = 0
    
    def minimum(self,l):
        self.min = round(l.min(),2)
    
    def maximum(self,l):
        self.max = round(l.max(),2)
    
    def Rng(self,l):
        self.range = round((l.max() - l.min()),2)
    
    def mn(self,l):
        self.mean = round(l.mean(),2)
    
    def mdn(self,l):
        self.median = round(l.median(),2)
    
    def stad(self,l):
        self.std = round(l.std(),2)
        
    def vre(self,l):
        self.variance = round(l.var(),2)
        
    def cal_stats(self,l):
        self.minimum(l)
        self.maximum(l)
        self.Rng(l)
        self.mn(l)
        self.mdn(l)
        self.stad(l)
        self.vre(l)
        
        return [self.min,self.max,self.range,self.mean,self.median,self.std,self.variance]


    
    