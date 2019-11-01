import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('50_Startups.csv')

dataset.head(5)
dataset.info()
dataset.describe()
dataset.select_dtypes(include=['float','int']).isnull().sum()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

sns.pairplot(dataset)
sns.distplot(dataset['R&D Spend'],kde=False, bins=30)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Le = LabelEncoder()
X[:,3] = Le.fit_transform(X[:,3])
onehot = OneHotEncoder(categorical_features=[3])
X = onehot.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test).reshape(-1,1)
y_test = y_test.reshape(-1,1)

from sklearn.metrics import mean_squared_error
print(regressor.score(X_test,y_test))
print(mean_squared_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))

import statsmodels.regression.linear_model as lm
X = np.append(arr= np.ones((50,1)).astype(float), values=X, axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_ols = lm.OLS(endog=y,  exog=X_opt).fit()
regressor_ols.summary()
regressor_ols.pvalues()

def backwardelimination(x,sl):
    numvar = len(x[1])
    for i in range(0,numvar):
        regressor_ols = lm.OLS(y,x).fit()
        maxpval = max(regressor_ols.pvalues).astype(float)

        if maxpval > sl:
            for j in range(0,numvar-i):
                if(regressor_ols.pvalues[j].astype(float) == maxpval):
                    x = np.delete(x,j,1)

    regressor_ols.summary()
    return x

X_opt = X[:,[0,1,2,3,4,5]]
SL = 0.5
X_new = backwardelimination(X_opt,SL)

# sns.scatterplot(X_new[:,2],y)
# sns.scatterplot(X_new[:,1], y)

regressor.score(X_train,y_train)
regressor.coef_
regressor.intercept_

