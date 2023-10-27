import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
churn=pd.read_csv('C:\\Users\\Danny\\archive\\Churn_Modelling.csv')
churn.head()
churn.shape
churn.size
churn.columns=churn.columns.str.strip()
churn.columns=churn.columns.str.lower()
churn.columns
churn.isnull().sum()
churn[churn.duplicated(subset=['customerid'],keep=False)]
churn.describe()
plt.figure(figsize=(15,5))
sns.countplot(data=churn,x='exited')
plt.figure(figsize=(15,5))
sns.countplot(data=churn,x='exited')
from sklearn.utils import resample
churn_majority=churn[churn['exited']==0]
churn_minority=churn[churn['exited']==1]
churn_majority_downsample=resample(churn_majority,n_samples=2037,replace=False,random_state=42)
churn_df=pd.concat([churn_majority_downsample,churn_minority])
churn_df['exited'].value_counts().to_frame()
plt.figure(figsize=(15,5))
sns.countplot(data=churn_df,x='exited')
churn_df.columns
churn_df.drop(['rownumber', 'customerid', 'surname','geography','gender'],axis=1,inplace=True)
churn_df.corr()
plt.figure(figsize=(15,5))
sns.heatmap(churn_df.corr(),annot=True)
df_corr_exit=churn_df.corr()['exited'].to_frame()
plt.figure(figsize=(15,5))
sns.barplot(data=df_corr_exit,x=df_corr_exit.index,y='exited')
x=churn_df.drop(['exited'],axis=1)
y=churn_df['exited']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=500)
lr.fit(x_train,y_train)
lr.score(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,f1_score,ConfusionMatrixDisplay
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)
cmd=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred,labels=lr.classes_),display_labels=lr.classes_)
cmd.plot()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
knn.score(x_test,y_test)
y_pred=lr.predict(x_test)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
svc.score(x_train,y_train)
svc.score(x_test,y_test)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)
cmd=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred,labels=svc.classes_),display_labels=svc.classes_)
cmd.plot()
