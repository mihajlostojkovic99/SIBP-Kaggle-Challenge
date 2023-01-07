import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)
data = pd.read_csv('datasets/Train_rev1.csv')
# data.drop("Id", axis=1, inplace=True)

print("First 5 rows:")
print(data.head(), end='\n \n')
print("Last 5 rows:")
print(data.tail(), end='\n \n')
print("Info about the dataset:")
print(data.info(), end='\n \n')
print("General statistic information about attributes:")
print(data.describe())
print("\nNumber of missing values for each column:")
print(data.isna().sum())

print("\nColumns with string values:")
for label, content in data.items():
    if pd.api.types.is_string_dtype(content):
        print(label)

print("\nColumns with numeric values:")
for label, content in data.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)

print("\n------------------------------ DATA ENGINEERING ------------------------------")

data_train = data.copy()

print("\nTurning all string values into unique number values...")
# Maybe we shouldn't use labelEncoder but pandas category type?
le = LabelEncoder()
for label, content in data_train.items():
    if pd.api.types.is_string_dtype(content):
        data_train[label] = le.fit_transform(data_train[label])

print("Generating IsMissing column for missing values...\n")
for label, content in data_train.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Create a new column for checking if data is missing
        data_train[label + "IsMissing"] = pd.isnull(content)
        # Transform categories into their respective codes and add 1
        # data_train[label] = pd.Categorical(content).codes + 1

# pd.set_option('display.max_columns', 22)
# print(data_train.head(10))

# print('\nCategory correlation with SalaryNormalized is ', data_train.SalaryNormalized.corr(data_train.Category),
#       ' which is what?')

print("\n------------------------------ TRAINING THE MODEL ------------------------------")

data_copy = data_train.copy()

X = data_copy.drop(columns=["SalaryNormalized"], axis=1)
y = data_copy["SalaryNormalized"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

print("\n--------- Random Forest Regression ---------")
print("\nTraining a regression model using Random Forest Regression...")

rfr_model = RandomForestRegressor(n_jobs=-1)
rfr_model.fit(X_train, y_train)
rfr_y_pred = rfr_model.predict(X_test)
mae_rand_forest = mean_absolute_error(y_test, rfr_y_pred)
mse_rand_forest = mean_squared_error(y_test, rfr_y_pred, squared=False)
r2_rand_forest = r2_score(y_test, rfr_y_pred)
print("Mean Absolute Error of this model is: ", mae_rand_forest)
print("Root Mean Square Error of this model is: ", mse_rand_forest)
print("R2 score of this model is [0..1]: ", r2_rand_forest)

print("\n--------- Linear Regression ---------")
print("\nTraining a regression model using Linear Regression...")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
mae_linear_regression = mean_absolute_error(y_test, lr_y_pred)
mse_linear_regression = mean_squared_error(y_test, lr_y_pred, squared=False)
r2_linear_regression = r2_score(y_test, lr_y_pred)
print("Mean Absolute Error of this model is: ", mae_linear_regression)
print("Root Mean Square Error of this model is: ", mse_linear_regression)
print("R2 score of this model is [0..1]: ", r2_linear_regression)
