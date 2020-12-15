# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:44:01 2017

@author: ASHIK
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing,cross_validation

def preprocess():
     df = pd.read_csv('F:/python/Anaconda/Python codes/Classification/KNN/breast-cancer-wisconsin.data.txt')
     df.replace('?', -99999, inplace=True)
     df.drop(['id'], 1, inplace= True)

     X = np.array(df.drop(['class'], 1))
     y = np.array(df['class'])

     X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.05)
     
     return X_train, X_test, y_train, y_test