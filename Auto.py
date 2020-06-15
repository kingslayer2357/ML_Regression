# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:23:22 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Regression\Auto.csv",header=None)

#Matrix and vector
X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Regressor
from sklearn.tree import DecisionTreeRegressor as DTR
regressor=DTR()
regressor.fit(X_train,y_train)


#prediction
y_pred=regressor.predict(X_test)


#Plotting
plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.show()



plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,y_pred,color="blue")
plt.show()