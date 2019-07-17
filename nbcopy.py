# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:44:03 2018

@author: DIVYASHREE R
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#load dataset
names=['class','mt1','mt5','mt10','mt1c','mt5c','doctorate','masters','bilingual','publications','skills','interned','gender']
dataset=pd.read_csv(r'C:/Users/miriam s/Documents/project_2018_dataset_draft1_csv_binary.csv',encoding="ISO-8859-1",header=None,names=names)
#dataset = pd.read_csv('LinkedinDS.csv',encoding='ISO-8859-1',header=None)
dataset = dataset.replace({'y':1,'n':0,'M':1,'F':0})
x = dataset.iloc[:,0].values #dependent variable --class--
Y = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]].values#data matrix
#print(dataset.shape,x.shape,Y.shape)

#Train - Test Splitting
#from sklearn.cross_validation import train_test_split
#from sklearn import model_selection
#x_train,x_test,Y_train,Y_test = train_test_split(x,Y,train_size = 0.8)
#print(x_train.shape,x_test.shape,Y_train.shape,Y_test.shape)

x_train,x_test,Y_train,Y_test=model_selection.train_test_split(x,Y,test_size=0.20,random_state=7)
from sklearn.naive_bayes import GaussianNB
seed=7
scoring='accuracy'
model=GaussianNB()
results=[]
kfold=model_selection.KFold(n_splits=10,random_state=seed)
cv_results=model_selection.cross_val_score(model,Y_train,x_train,cv=kfold,scoring=scoring)
results.append(cv_results)
msg="GNB: %f (%f)"%(cv_results.mean(),cv_results.std())
gnb = GaussianNB()
gnb.fit(Y_train,x_train)
expected1 = x_test
predicted1 = gnb.predict(Y_test)

from sklearn.naive_bayes import BernoulliNB
seed=7
scoring='accuracy'
model=BernoulliNB()
results=[]
kfold=model_selection.KFold(n_splits=10,random_state=seed)
cv_results=model_selection.cross_val_score(model,Y_train,x_train,cv=kfold,scoring=scoring)
results.append(cv_results)
msg1="BNB: %f (%f)"%(cv_results.mean(),cv_results.std())
bnb = BernoulliNB()
bnb.fit(Y_train,x_train)
expected2 = x_test
predicted2 = bnb.predict(Y_test)

inp1=int(input("Do you have more than 1 year of experience? 1 for yes or 0 for no: "))
inp2=int(input("Do you have more than 5 years of experience? 1 for yes or 0 for no: "))
inp3=int(input("Do you have more than 10 years of experience? 1 for yes or 0 for no: "))
inp4=int(input("Do you have more than 1 year of experience in current company? 1 for yes or 0 for no: "))
inp5=int(input("Do you have more than 5 years of experience in current company? 1 for yes or 0 for no: "))
inp6=int(input("Do you have a doctorate? 1 for yes or 0 for no: "))
inp7=int(input("Do you have a masters degree? 1 for yes or 0 for no: "))
inp8=int(input("Are you multilinguistic? 1 for yes or 0 for no: "))
inp9=int(input("Do you have any publications/patents? 1 for yes or 0 for no: "))
inp10=int(input("Do you possess more than 20 skills? 1 for yes or 0 for no: "))
inp11=int(input("Have you interned? 1 for yes or 0 for no: "))
inp12=int(input("Your gender? 1 for Male or 0 for Female: "))
from sklearn import metrics
print("GNB")
print(msg)
print(accuracy_score(x_test,predicted1))
print(metrics.classification_report(expected1,predicted1))
print(metrics.confusion_matrix(expected1,predicted1))
#print(gnb.predict([[inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,inp11,inp12]]))
print("BNB")
print(msg1)
print(accuracy_score(x_test,predicted2))
print(metrics.classification_report(expected2,predicted2))
print(metrics.confusion_matrix(expected2,predicted2))
print("The predicted class of company is: ")
print(bnb.predict([[inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,inp11,inp12]]))