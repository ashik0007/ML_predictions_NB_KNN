# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:30:57 2018

@author: ASHIK
"""

import numpy as np
import pandas as pd
from sklearn import svm,cross_validation

df = pd.read_csv('semeion - text.txt' , sep = '\s+' , dtype=int)
     
data = np.array(df)
X = []
y = []

i = 0
while i<1592:
     data_X = data[i][:256]
     data_y = data[i][256:266]
     
     X.extend(data_X)
     y.extend(data_y)
     i+=1
     
X_train_1 = np.array(X)
y_train_1 = np.array(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_train_1, y_train_1, test_size=0.2)
  
clf = svm.SVC()
clf.fit(X_train , y_train)
accuracy = clf.score(X_test , y_test)
print(accuracy)

     
     
