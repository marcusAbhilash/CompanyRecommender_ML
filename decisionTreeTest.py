import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
names=['class','mt1','mt5','mt10','mt1c','mt5c','doctorate','masters','bilingual','publications','skills','interned','gender']
dataset=pandas.read_csv(r'C:/Users/miriam s/Documents/project_2018_dataset_draft1_csv_binary.csv',encoding="ISO-8859-1",header=None,names=names)
dataset = dataset.replace({'y':1,'n':0,'M':1,'F':0})
array=dataset.values
X=array[:,[1,2,3,4,5,6,7,8,9,10,11,12]]
Y=array[:,0]
validation_size=0.20
seed=7
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=7)
seed=7
scoring='accuracy'
model=DecisionTreeClassifier()
results=[]
kfold=model_selection.KFold(n_splits=10,random_state=seed)
cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
results.append(cv_results)
msg="DecisionTree: %f (%f)"%(cv_results.mean(),cv_results.std())
print(msg)
dt=DecisionTreeClassifier(criterion="gini",random_state=seed,max_depth=None)
dt.fit(X_train,Y_train)
predictions=dt.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))
print(dt.predict([[1,0,0,1,0,0,0,1,0,0,0,1]]))