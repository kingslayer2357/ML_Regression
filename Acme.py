# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:02:12 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Regression\Acme.csv")

#matrix and vector
X=dataset.iloc[:,2].values
X=np.reshape(X,(-1,1))

y=dataset.iloc[:,3].values

#training and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

#Regressor
from sklearn.ensemble import DecisionTreeRegressor as DTR
regressor=DTR()
regressor.fit(X_train,y_train)


#predictions
y_pred=regressor.predict(X_test)

#plot
plt.scatter(X_test,y_test)
plt.plot(X_test,regressor.predict(X_test))