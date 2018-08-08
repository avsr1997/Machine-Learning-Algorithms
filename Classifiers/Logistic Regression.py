import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Social_Network_Ads.csv')

x=dataset.iloc[:,3:4].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
from sklearn.metrics import accuracy_score
ascore=accuracy_score(y_test,y_predict)

X=dataset.iloc[:,2:4].values
Y=dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/4,random_state=0)

from sklearn.linear_model import LogisticRegression
clf1=LogisticRegression()
clf1.fit(X_train,Y_train)
Y_predict=clf1.predict(X_test)

from sklearn.metrics import confusion_matrix
CM=confusion_matrix(Y_test,Y_predict)
from sklearn.metrics import accuracy_score
ASCORE=accuracy_score(Y_test,Y_predict)

from matplotlib.colors import ListedColormap
X_set,Y_set=X_train,Y_train
X1,X2=np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1,X2,clf1.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.scatter(X_set[Y_set==0,0],X_set[Y_set==0,1],color='red')
plt.scatter(X_set[Y_set==1,0],X_set[Y_set==1,1],color='green')

from matplotlib.colors import ListedColormap
X_set,Y_set=X_test,Y_test
X1,X2=np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1,X2,clf1.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.scatter(X_set[Y_set==0,0],X_set[Y_set==0,1],color='red')
plt.scatter(X_set[Y_set==1,0],X_set[Y_set==1,1],color='green')