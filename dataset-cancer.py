from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

class classifier:
    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
    def predict(self,x_test):
        prediction2=[]
        for i in len(x_test):
            for j in len(x_train):
                best=smallest(x_test[i],x_train[j])
                best_index=0
                if(distance.euclidean(x_test[i],x_train[j])<=best):
                    best=distance.euclidean(x_test[i],x_train[j])
                    best_index=y_train[j]
            prediction2.append(y_train[best_index])
        return prediction2

def smallest(a,b):
    return distance.euclidean(a,b)
        
cancer=load_breast_cancer()
print(cancer.data.shape)
print(cancer.data)
print(cancer.target)
print(cancer.feature_names)
print(cancer.target_names)

for i in range(0,569):
    for j in range(0,569):
        if(cancer.target[j]<cancer.target[i]):
            temp1=cancer.data[j]
            cancer.data[j]=cancer.data[i]
            cancer.data[i]=temp1
            temp2=cancer.target[j]
            cancer.target[j]=cancer.target[i]
            cancer.target[i]=temp2

print(cancer.target)

i=0
while(i<369):
    print(cancer.data[i],"*****",cancer.target[i])
    i +=1 
   
x=cancer.data
y=cancer.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)
normal_classifier=KNeighborsClassifier()
normal_classifier.fit(x_train,y_train)
predictions1=normal_classifier.predict(x_test)
print(accuracy_score(y_test,predictions1))

myclassifier=classifier()
myclassifier.fit(x_train,y_train)
predictions2=myclassifier.predict(x_test)