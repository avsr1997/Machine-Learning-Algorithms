#Sample Program to Train Classifier on Iris Dataset
from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np

# Test data is created in a list
test_id=[0,50,100]
iris=load_iris()
train_data=np.delete(iris.data,test_id,axis=0)
train_target=np.delete(iris.target,test_id)
test_data=iris.data[test_id]
test_target=iris.target[test_id]

# Tree classifier is used to train the whole dataset
clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

#Generating a pdf file of Decision Tree on Wine Dataset
from sklearn.externals.six import StringIO
import pydotplus
dot_data=StringIO()
tree.export_graphviz(clf,out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,rounded=True,
                     impurity=False)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_tree.pdf")                     
