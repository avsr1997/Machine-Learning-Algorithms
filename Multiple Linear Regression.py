import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/5)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_predict=regressor.predict(X_test)

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]

from statsmodels.formula.api import OLS
ols_regressor=OLS(endog=Y,exog=X_opt).fit()
ols_regressor.summary()

X_opt=X[:,[0,1,3,4,5]]
ols_regressor=OLS(endog=Y,exog=X_opt).fit()
ols_regressor.summary()

X_opt=X[:,[0,3,4,5]]
ols_regressor=OLS(endog=Y,exog=X_opt).fit()
ols_regressor.summary()

X_opt=X[:,[0,3,5]]
ols_regressor=OLS(endog=Y,exog=X_opt).fit()
ols_regressor.summary()

X_train,X_test,Y_train,Y_test=train_test_split(X_opt,Y,test_size=1/5)
regressor.fit(X_train,Y_train)
Y_final=regressor.predict(X_test)