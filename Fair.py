# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 02:51:29 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Regression\Fair.csv")

#Matrix and vector
X=dataset.iloc[:,[1,2,3,5,6,7,8,9]].values
y=dataset.iloc[:,4].values

#Encoding categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder=LabelEncoder()
X[:,0]=encoder.fit_transform(X[:,0])
encoder2=LabelEncoder()
y=encoder2.fit_transform(y)

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)




#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
var=pca.explained_variance_ratio_

#Classifier
from sklearn.linear_model import LogisticRegression as LR
regressor=LR()
regressor.fit(X_train,y_train)


#prediction
y_pred=regressor.predict(X_test)

#cm
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




#Appling k fold
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressor,X=X_train,y=y_train,cv=10)
m=accuracies.mean()


#plot
plt.scatter(X_train,y_train,color="red")
plt.show()