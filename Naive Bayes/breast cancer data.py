# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 23:42:07 2017

@author: ASHIK
"""

from sklearn import naive_bayes
from breast_cancer_preprocessing import preprocess

X_train, X_test, y_train, y_test = preprocess()

clf = naive_bayes.GaussianNB()
clf.fit(X_train,y_train)

print('Training is done.')
print('Testing is done.\n')

print('Your prediction is finished.')
pred = clf.predict(X_test)
print(pred)
print('\n')

accuracy = clf.score(X_test,y_test)
print('The accuracy is: ',accuracy)
print('\n')


