import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
linear_regressor1=LinearRegression()
linear_regressor1.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,linear_regressor1.predict(X),color='blue')
plt.title('Polynomial Study')
plt.xlabel('Post')
plt.ylabel('Salary')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=4)
X_poly=poly_features.fit_transform(X)
linear_regressor2=LinearRegression()
linear_regressor2.fit(X_poly,Y)

X_perfect=np.arange(min(X),max(X),0.1)
X_perfect=X_perfect.reshape(len(X_perfect),1)

plt.scatter(X,Y,color='red')
plt.plot(X_perfect,linear_regressor2.predict(poly_features.fit_transform(X_perfect)),color='blue')
plt.title('Polynomial Study')
plt.xlabel('Post')
plt.ylabel('Salary')
plt.show()