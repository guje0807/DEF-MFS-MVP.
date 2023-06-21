# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:29:58 2023

@author: aakas"""

import matplotlib.pyplot as plt

class viz:
    def line_chart(self,a,b,d):
        plt.figure(figsize=(10,6))
        plt.plot(d,a,label="Ford")
        plt.plot(d,b,label='TSLA')
        plt.legend()
        plt.show()