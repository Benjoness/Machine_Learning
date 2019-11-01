import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('advertising.csv')

dataset.head(5)
dataset.info()
dataset.describe()

sns.set_style('whitegrid')
dataset['Age'].hist(bins=30)
plt.xlabel('Age')

sns.jointplot(x='Age',y='Area Income',data=dataset)
sns.pairplot(dataset,hue='Clicked on Ad', palette='bwr')

X = dataset.iloc[:,[0,1,2,3,6]].values
y = dataset.iloc[:,-1].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_pred))
print('\n')
print(confusion_matrix(y_test,y_pred))

