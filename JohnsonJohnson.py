# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:35:33 2020

@author: kingslayer
"""


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Regression\JohnsonJohnson.csv")

#Matrix and vector
X=dataset.iloc[:,1].values
y=dataset.iloc[:,2].values
X=np.reshape(X,(-1,1))

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures()
X_poly=poly_reg.fit_transform(X)

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_poly,y,test_size=0.2,random_state=0)




#Regressor
from sklearn.linear_model import LinearRegression

regressor=LinearRegression(degree=2)
regressor.fit(X_train,y_train)


#prediction
y_pred=regressor.predict(X_test)


#Plotting
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.show()


plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,y_pred,color="blue")
plt.show()