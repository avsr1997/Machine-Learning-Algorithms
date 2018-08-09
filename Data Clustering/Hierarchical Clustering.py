import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
graph=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Hierarchical Clustering')
plt.xlabel('Dataset')
plt.ylabel('Euclidean Distance')

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_predict=hc.fit_predict(x)

plt.scatter(x[y_predict==0,0],x[y_predict==0,1],color='black')
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],color='green')
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],color='red')
plt.scatter(x[y_predict==3,0],x[y_predict==3,1],color='blue')
plt.scatter(x[y_predict==4,0],x[y_predict==4,1],color='orange')