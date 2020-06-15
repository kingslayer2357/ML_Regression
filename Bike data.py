# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:17:27 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Regression\Bike data.csv")

#Matrix and vector
X=dataset.iloc[:,1:13].values
y=dataset.iloc[:,13].values

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
#var=pca.explained_variance_ratio_


#Regressor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


#prediction
y_pred=regressor.predict(X_test)


#Plotting
plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.show()
