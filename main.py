from linear_models.linear_regression import LinearRegression
from linear_models.ridge import Ridge
from linear_models.lasso import Lasso
from utils.model_selection import train_test_split
from utils.preprocessing import StandardScaler
from utils.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso

data = fetch_california_housing()

# X, y = datasets.make_regression(n_samples=100, n_features=300, noise=100, random_state=42)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()

lr.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

predictions_lr = lr.predict(X_test)
predictions_ridge = ridge.predict(X_test)
predictions_lasso = lasso.predict(X_test)

print("Linear Regression")
print("MSE", mean_squared_error(y_test, predictions_lr))
print("MAE:", mean_absolute_error(y_test, predictions_lr))
print("RMSE:", root_mean_squared_error(y_test, predictions_lr))
print("R2:", r2_score(y_test, predictions_lr))
print("------")

print("Ridge")
print("MSE", mean_squared_error(y_test, predictions_ridge))
print("MAE:", mean_absolute_error(y_test, predictions_ridge))
print("RMSE:", root_mean_squared_error(y_test, predictions_ridge))
print("R2:", r2_score(y_test, predictions_ridge))
print("------")

print("Lasso")
print("MSE", mean_squared_error(y_test, predictions_lasso))
print("MAE:", mean_absolute_error(y_test, predictions_lasso))
print("RMSE:", root_mean_squared_error(y_test, predictions_lasso))
print("R2:", r2_score(y_test, predictions_lasso))
print("------")