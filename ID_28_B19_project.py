import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('GDSC2_dataset.csv')

X = dataset.iloc[:, :-1]  
y = dataset.iloc[:, -1]    

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[numerical_cols] = imputer.fit_transform(X[numerical_cols])

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X, y)

support = selector.get_support()
feature_names = X.columns[support]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.scatter(y_train, y_pred_train, color='red')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='black', linestyle='--')
plt.title('Training Set')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_test, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--')
plt.title('Test Set')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()