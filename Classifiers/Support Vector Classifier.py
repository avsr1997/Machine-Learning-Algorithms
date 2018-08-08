import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,2:4].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

from sklearn.svm import SVC
clf=SVC(kernel='linear',random_state=0)
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_predict)
ascore=accuracy_score(y_test,y_predict)

from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
X,Y=np.meshgrid(np.arange(x_set[:,0].min() - 1,x_set[:,0].max() + 1,0.01),
                np.arange(x_set[:,1].min() - 1,x_set[:,1].max() + 1,0.01))
plt.contourf(X,Y,clf.predict(np.array([X.ravel(),Y.ravel()]).T).reshape(X.shape),
             alpha=0.5,cmap=ListedColormap(('red','green')))
plt.scatter(x_set[y_set==0,0],x_set[y_set==0,1],color='red')
plt.scatter(x_set[y_set==1,0],x_set[y_set==1,1],color='green')

from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
X,Y=np.meshgrid(np.arange(x_set[:,0].min() - 1,x_set[:,0].max() + 1,0.01),
                np.arange(x_set[:,1].min() - 1,x_set[:,1].max() + 1,0.01))
plt.contourf(X,Y,clf.predict(np.array([X.ravel(),Y.ravel()]).T).reshape(X.shape),
             alpha=0.5,cmap=ListedColormap(('red','green')))
plt.scatter(x_set[y_set==0,0],x_set[y_set==0,1],color='red')
plt.scatter(x_set[y_set==1,0],x_set[y_set==1,1],color='green')