import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
digit=load_digits()
myclassifier=KNeighborsClassifier()

test_id=[]
for i in range(0,1797,175):
    test_id.append(i)        
print("test_id",test_id)

train_data=np.delete(digit.data,test_id,axis=0)
train_target=np.delete(digit.target,test_id)
print(train_data)
print(train_target)
print(len(train_data))
print(len(train_target))

test_data=digit.data[test_id]
test_target=digit.target[test_id]
print(test_data)
print(test_target)
print(len(test_data))
print(len(test_target))

myclassifier.fit(train_data,train_target)
myclassifier.predict(test_data)
print("Test Results:")
print(myclassifier.predict(test_data))


import matplotlib.pyplot as plt
for i in range(1,10):
    plt.gray()
    plt.matshow(digit.images[i])
    plt.show()