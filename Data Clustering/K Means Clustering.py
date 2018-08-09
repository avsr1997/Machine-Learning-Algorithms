import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values



from sklearn.cluster import KMeans
total=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    total.append(kmeans.inertia_)
plt.plot(range(1,11),total)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Sum Square')
plt.show()

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
kmeans.fit(x)
y_predict=kmeans.predict(x)

plt.scatter(x[y_predict==0,0],x[y_predict==0,1],color='red',label='Cluster 1')
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],color='blue',label='Cluster 2')
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],color='black',label='Cluster 3')
plt.scatter(x[y_predict==3,0],x[y_predict==3,1],color='green',label='Cluster 4')
plt.scatter(x[y_predict==4,0],x[y_predict==4,1],color='yellow',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='orange')