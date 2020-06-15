# -*- coding: utf-8 -*-
"""
Created on Fri May  1 04:02:26 2020

@author: kingslayer
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  1 03:57:08 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Regression\Treer.csv")

#Matrix and vector
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
var=pca.explained_variance_ratio_






#Regressor
from sklearn.ensemble import RandomForestRegressor as RFR

regressor=RFR(n_estimators=1000)
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


