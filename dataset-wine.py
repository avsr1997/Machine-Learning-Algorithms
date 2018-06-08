# Sample Program to Train Classifier on Wine Dataset
from sklearn import tree
from sklearn.datasets import load_wine
import numpy as np

# Test data is created in list
test_id=[10,100,170]
wine=load_wine()
train_data=np.delete(wine.data,test_id,axis=0)
train_target=np.delete(wine.target,test_id)
test_data=wine.data[test_id]
test_target=wine.target[test_id]

# Tree classifier is used to train thw dataset
clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)
print(clf.predict(test_data))

# Generating a pdf file of Decision Tree on Wine Dataset
from sklearn.externals.six import StringIO
import pydotplus
dot_data=StringIO()
tree.export_graphviz(clf,out_file=dot_data,
                     feature_names=wine.feature_names,
                     class_names=wine.target_names,
                     filled=True,rounded=True,
                     impurity=False)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("wine_tree.pdf")

# Training on KNearestNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
myclassifier=KNeighborsClassifier()
myclassifier.fit(train_data,train_target)
myclassifier.predict(test_data)
