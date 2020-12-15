# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:15:44 2017

@author: ASHIK
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation,svm
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X, y, test_size=0.2)

'''
test_size = 0.1
kf = cross_validation.StratifiedKFold(y, round(1/test_size))

train_indices, valid_indices = next(iter(kf))

X_train, y_train = X[train_indices], y[train_indices]
X_valid, y_valid = X[valid_indices], y[valid_indices]

'''
clf = GaussianNB()
clf.fit(X_train,y_train)

accuracy =clf.score(X_valid,y_valid)
print(accuracy)

