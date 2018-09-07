# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 02:05:58 2018

@author: karale tejas pradip
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
#Importing dataset
dataset=pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
X_test = X_test.iloc[:,1:-2].values
X_train = dataset.iloc[:, 1:-3].values
y_train = dataset.iloc[:, 17].values
X = np.concatenate((X_train,X_test))
X = np.delete(X[:,:],X[:,9],axis=1)
temp1 = DataFrame(data=X)


#filling values in column of credit score

for i in range(105238):
    if X[:,2][i]<5000:
        X[:,2][i]=0
    elif pd.isnull(X[:,2][i]):
        X[:,2][i]=0
    else:
        X[:,2][i]=1

#Filling missing values
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,[5]])
X[:,[5]]=imputer.transform(X[:,[5]])






for i in range(105238):
    if pd.isnull(X[:,3][i]):
        X[:,3][i]="5 years"
 '''   
imputer = Imputer(missing_values = 'NaN',strategy='most_frequent')        
imputer = imputer.fit(X[:,[13,14]])
X[:,[13,14]] = imputer.transform(X[:,[13,14]])    
''' 
'''       
imputer = Imputer(missing_values = 'NaN',strategy='median')        
imputer = imputer.fit(X[:,[9]])
X[:,[9]] = imputer.transform(X[:,[9]])
'''



#encoding categorial variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
cat_var=[1,3,4,6]
for i in cat_var:
    X[:,i]=LabelEncoder().fit_transform(X[:,i])

onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
X = X[:,1:] 

onehotencoder=OneHotEncoder(categorical_features=[13])
X=onehotencoder.fit_transform(X).toarray()
X = X[:,1:]    

onehotencoder=OneHotEncoder(categorical_features=[17])
X=onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)




X_train1 = X[0:84190,:]
X_test1 = X[84190:,:]

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train1, y_train)

y_pred = classifier.predict(X_test1)

sub = pd.read_csv("sample_submission.csv")

sub["Predicted"] = y_pred

sub.to_csv("submission_14.csv" , index = False)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train1, y_train, test_size = 0.2, random_state = 0)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train2, y_train2)

y_pred2 = classifier.predict(X_test2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2,y_pred2) 








