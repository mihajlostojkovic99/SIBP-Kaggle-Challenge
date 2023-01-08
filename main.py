import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
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

params_grid = {
    "n_estimators": np.arange(50, 200, 10),
    # "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
    "max_features": [0.5, 1, "sqrt", "auto"],
    "max_samples": [None, 10000, 12000, 15000, 20000]
}

models = {
    'Linear Regression': LinearRegression(n_jobs=-1),
    'Random Forest': RandomForestRegressor(n_jobs=-1, verbose=5),
    # 'Random Forest (absolute error criterion)': RandomForestRegressor(n_jobs=-1, criterion="absolute_error"),  # death
    'Random Forest hyperparameter tuned': RandomizedSearchCV(RandomForestRegressor(n_jobs=-1, warm_start=True),
                                                             param_distributions=params_grid, n_iter=50, verbose=5,
                                                             n_jobs=-1),
    'XGBoost': XGBRegressor()
}

results = pd.DataFrame(columns=['MAE', 'RMSE', 'R2-score'])

for model, func in models.items():
    print("\nTraining a ", model, "model...")
    func.fit(X_train, y_train)
    pred = func.predict(X_test)
    results.loc[model] = [mean_absolute_error(y_test, pred),
                          mean_squared_error(y_test, pred, squared=False),
                          r2_score(y_test, pred)
                          ]

print("\n\nBest Random Forest hyperparameters which achieved a score of ",
      models['Random Forest hyperparameter tuned'].best_score_, "were: ",
      models['Random Forest hyperparameter tuned'].best_params_)

print("\n", results)
