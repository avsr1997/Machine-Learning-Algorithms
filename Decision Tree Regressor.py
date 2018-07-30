import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

X_perfect=np.arange(min(X),max(X),0.1)
X_perfect=X_perfect.reshape(len(X_perfect),1)

plt.scatter(X,Y,color='red')
plt.plot(X_perfect,regressor.predict(X_perfect),color='blue')
plt.title('Decision Tree Regressor')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

Y_predict=regressor.predict(6.5)