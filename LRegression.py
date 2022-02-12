import numpy as np
import pandas as pd

companies = pd.read_csv('/usr/src/app/1000_Companies.csv')
X = companies.iloc[:, :-1].values
y = companies.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_test=pd.DataFrame(y_test, columns=['Original'])
y_test['Prediction']=y_pred

y_test.to_csv('/usr/src/app/Result.csv',index=False)
