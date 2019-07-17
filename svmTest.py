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
model=SVC(kernel='rbf',C=1,gamma=0.5)
results=[]
kfold=model_selection.KFold(n_splits=10,random_state=seed)
cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
results.append(cv_results)
msg="SVM: %f (%f)"%(cv_results.mean(),cv_results.std())
#svc=svm.SVC(kernel='rbf',C=1,gamma=0.5).fit(X,Y)
svc=svm.SVC(kernel='linear',C=1,gamma='auto').fit(X,Y)
svc.fit(X_train,Y_train)
predictions=svc.predict(X_validation)
inp1=input("Do you have more than 1 year of experience? 1 for yes or 0 for no: ")
inp2=input("Do you have more than 5 years of experience? 1 for yes or 0 for no: ")
inp3=input("Do you have more than 10 years of experience? 1 for yes or 0 for no: ")
inp4=input("Do you have more than 1 year of experience in current company? 1 for yes or 0 for no: ")
inp5=input("Do you have more than 5 years of experience in current company? 1 for yes or 0 for no: ")
inp6=input("Do you have a doctorate? 1 for yes or 0 for no: ")
inp7=input("Do you have a masters degree? 1 for yes or 0 for no: ")
inp8=input("Are you multilinguistic? 1 for yes or 0 for no: ")
inp9=input("Do you have any publications/patents? 1 for yes or 0 for no: ")
inp10=input("Do you possess more than 20 skills? 1 for yes or 0 for no: ")
inp11=input("Have you interned? 1 for yes or 0 for no: ")
inp12=input("Your gender? 1 for Male or 0 for Female: ")
print(msg)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

print("The predicted class of company is: ")
print(svc.predict([[inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8,inp9,inp10,inp11,inp12]]))