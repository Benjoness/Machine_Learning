import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Salary_Data.csv')

dataset.head(5)
dataset.info()
dataset.describe()

sns.distplot(dataset['Salary'], kde=False)
sns.scatterplot(x=dataset['YearsExperience'], y=dataset['Salary'])

X = dataset.iloc[:,0].values.reshape(-1,1)
y = dataset.iloc[:,1].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model  import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

y_pred = lm.predict(X_test)

from sklearn.metrics import mean_squared_error
print(lm.score(X_test,y_test))
print(mean_squared_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))

# sns.scatterplot(x=X_train[:,0], y=y_train[:,0])
plt.scatter(X_train,y_train)
plt.plot(X_train, lm.predict(X_train), color='red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lm.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

