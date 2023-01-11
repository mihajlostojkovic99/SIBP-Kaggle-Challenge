import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

print("\n------------------------------ PLOTTING AND DATA ANALYSIS ------------------------------")

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

plt.figure(figsize=(8, 6))
plt.scatter(range(data.shape[0]), np.sort(data.SalaryNormalized.values))
plt.xlabel('Id', fontsize=12)
plt.ylabel('SalaryNormalized', fontsize=12)
plt.title('Scattered SalaryNormalized values', fontdict={'fontsize': 18}, pad=16)
plt.savefig('Scattered Salary.png', dpi=250)
plt.show()

ulimit = np.percentile(data.SalaryNormalized.values, 99)
llimit = np.percentile(data.SalaryNormalized.values, 1)
data = data[(data["SalaryNormalized"] < ulimit) & (data["SalaryNormalized"] > llimit)]

plt.figure(figsize=(8, 6))
plt.scatter(range(data.shape[0]), np.sort(data.SalaryNormalized.values))
plt.xlabel('Id', fontsize=12)
plt.ylabel('SalaryNormalized', fontsize=12)
plt.title('Scattered SalaryNormalized values after removing outliers', fontdict={'fontsize': 18}, pad=16)
plt.savefig('Scattered Salary Removed Outliers.png', dpi=250)
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(data.SalaryNormalized.values, bins=50, kde=True)
plt.xlabel('SalaryNormalized', fontsize=12)
plt.title('Histogram plot of Salary', fontdict={'fontsize': 18}, pad=16)
plt.savefig('Histogram Salary.png', dpi=250)
plt.show()

cnt_category = data.Category.value_counts()
plt.figure(figsize=(16, 10))
sns.barplot(x=cnt_category.index, y=cnt_category.values, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel('Job Category', fontsize=12)
plt.ylabel('Number of Listings', fontsize=12)
plt.tight_layout()
plt.savefig('Listing Categories.png', dpi=250)
plt.show()

cnt_locations = data['LocationNormalized'].value_counts().nlargest(20)
plt.figure(figsize=(16, 10))
sns.barplot(x=cnt_locations.index, y=cnt_locations.values, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel('Location', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.title('20 Most Common Locations', fontdict={'fontsize': 18}, pad=16)
plt.tight_layout()
plt.savefig('Top 20 Locations.png', dpi=250)
plt.show()

print("\n------------------------------ DATA ENGINEERING ------------------------------")

data_train = data.copy()

print("\nGenerating IsMissing column for missing values...")
for label, content in data_train.items():
    if not pd.api.types.is_numeric_dtype(content):
        data_train[label + "IsMissing"] = pd.isnull(content)

data_train.drop(['FullDescriptionIsMissing', 'LocationRawIsMissing', 'LocationNormalizedIsMissing', 'CategoryIsMissing',
                 'SalaryRawIsMissing', 'SourceNameIsMissing'], inplace=True, axis=1)

print("Turning all string values into unique number values...")
le = LabelEncoder()
for label, content in data_train.items():
    if pd.api.types.is_string_dtype(content):
        data_train[label] = le.fit_transform(data_train[label]) + 1

# pd.set_option('display.max_columns', 22)
# print(data_train)

print("Plotting heatmap...")
plt.figure(figsize=(20, 14))
mask = np.triu(np.ones_like(data_train.corr(), dtype=bool))
heatmap = sns.heatmap(data_train.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)
plt.tight_layout()
plt.savefig('Correlation Heatmap.png', dpi=250)
plt.show()

# exit() # Uncomment to skip training and only perform analysis

print("\n------------------------------ TRAINING THE MODEL ------------------------------")

data_copy = data_train.copy()

X = data_copy.drop(columns=["SalaryNormalized"], axis=1)
y = data_copy["SalaryNormalized"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

lr_params_grid = {
    "fit_intercept": [True, False],
    "positive": [True, False]
}

rf_params_grid = {
    "n_estimators": np.arange(50, 200, 10),
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
    "max_features": [0.5, 1, "sqrt", "auto"],
    "max_samples": [None, 10000, 12000, 15000, 20000]
}

models = {
    'Tuned Linear Regression': GridSearchCV(LinearRegression(n_jobs=-1), lr_params_grid, n_jobs=-1),
    'Random Forest': RandomForestRegressor(n_jobs=-1, verbose=1),
    # 'Random Forest (absolute error criterion)': RandomForestRegressor(n_jobs=-1, criterion="absolute_error"),  # death
    'Hyperparameter Tuned Random Forest': RandomizedSearchCV(RandomForestRegressor(n_jobs=-1, warm_start=True),
                                                             param_distributions=rf_params_grid, n_iter=5, verbose=5,
                                                             n_jobs=-1),
    # Adjust n_iter for Random Forest based on how you value your time
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

if 'Tuned Linear Regression' in models.keys():
    print("\n\nBest Linear Regression hyperparameters which achieved a score of",
          models['Tuned Linear Regression'].best_score_, "were:",
          models['Tuned Linear Regression'].best_params_)

if 'Hyperparameter Tuned Random Forest' in models.keys():
    print("\n\nBest Random Forest hyperparameters which achieved a score of ",
          models['Random Forest hyperparameter tuned'].best_score_, "were: ",
          models['Random Forest hyperparameter tuned'].best_params_)

print("\n", results)
