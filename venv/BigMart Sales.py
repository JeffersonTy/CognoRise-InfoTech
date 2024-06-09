import pandas as pd
import numpy as np

from sklearn import datasets
# Encoding the Categorical Features
from sklearn.preprocessing import LabelEncoder
# Data Standardization
from sklearn.preprocessing import StandardScaler
# Train-Test Split
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# Model Comparison : Training & Evaluation
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from six.moves import urllib

warnings.filterwarnings("ignore")

# Loading data
big_mart_sales_df = pd.read_csv('Dataset/BigMart sales data/Train.csv')

print('The size of Dataframe is: ', big_mart_sales_df.shape)
print('-'*100)

print('The Column Name, Record Count and Data Types are as follows: ')
big_mart_sales_df.info()
print('-'*100)

print('Statistical information of the dataframe: ')
print(big_mart_sales_df.describe())
print('-'*100)

# Defining numerical & categorical columns
numeric_features = [feature for feature in big_mart_sales_df.columns if big_mart_sales_df[feature].dtype != 'O']
categorical_features = [feature for feature in big_mart_sales_df.columns if big_mart_sales_df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))


print('Amount and percentage of missing values in Dataframe: ')
print('-'*100)
total=big_mart_sales_df.isnull().sum().sort_values(ascending=False)
percent=(big_mart_sales_df.isnull().sum()/big_mart_sales_df.isnull().count()*100).sort_values(ascending=False)
final = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(final)


print('Summary Statistics of numerical features for DataFrame are as follows:')
print('-'*100)
print(big_mart_sales_df.describe())

print('Summary Statistics of categorical features for DataFrame are as follows:')
print('-'*100)
print(big_mart_sales_df.describe(include='object').T)
print('-'*100)


# Data Cleaning & Preprocessing
# Handling Missing Values

# Filling the missing values in "Item_weight column" with "Mean" value
big_mart_sales_df['Item_Weight'].fillna(big_mart_sales_df['Item_Weight'].mean(), inplace=True)

# Filling the missing values in "Outlet_Size" column with Mode
mode_of_Outlet_size = big_mart_sales_df['Outlet_Size'].mode()[0]
print(mode_of_Outlet_size)

big_mart_sales_df['Outlet_Size'].fillna(mode_of_Outlet_size, inplace=True)


print('Amount and percentage of missing values in Dataframe: ')
print('-'*100)
total=big_mart_sales_df.isnull().sum().sort_values(ascending=False)
percent=(big_mart_sales_df.isnull().sum()/big_mart_sales_df.isnull().count()*100).sort_values(ascending=False)
final2 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(final2)
print('-'*100)

# Graphical visualization of data distribution

def distplotchart(column:str):
    plt.figure(figsize=(6, 6))
    sns.distplot(big_mart_sales_df[column])
    plt.show()

def countplotchart(column:str):
    plt.figure(figsize=(6, 6))
    sns.countplot(x=column, data=big_mart_sales_df, palette='cubehelix')
    plt.show()

# Item weight distribution
distplotchart('Item_Weight')

# Item Visibility Distribution
distplotchart('Item_Visibility')

# Item MRP Distribution
distplotchart('Item_MRP')

# Item Outlet Sales Distribution
distplotchart('Item_Outlet_Sales')

# Outlet Establishment Year
countplotchart('Outlet_Establishment_Year')

# Item Fat Content Distribution
countplotchart('Item_Fat_Content')
#
# # Item Type Distribution
plt.figure(figsize=(15,15))
sns.countplot(x='Item_Type', data=big_mart_sales_df, palette='cubehelix')
plt.xticks(rotation=90)
plt.show()
#
# Outlet Size Distribution
countplotchart('Outlet_Size')


# Ensuring data uniformity
print(big_mart_sales_df)

print(big_mart_sales_df['Item_Fat_Content'].value_counts())
big_mart_sales_df.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
print(big_mart_sales_df['Item_Fat_Content'].value_counts())


# Encoding the Categorical Features
encoder = LabelEncoder()

big_mart_sales_df['Item_Identifier'] = encoder.fit_transform(big_mart_sales_df['Item_Identifier'])
big_mart_sales_df['Item_Fat_Content'] = encoder.fit_transform(big_mart_sales_df['Item_Fat_Content'])
big_mart_sales_df['Item_Type'] = encoder.fit_transform(big_mart_sales_df['Item_Type'])
big_mart_sales_df['Outlet_Identifier'] = encoder.fit_transform(big_mart_sales_df['Outlet_Identifier'])
big_mart_sales_df['Outlet_Size'] = encoder.fit_transform(big_mart_sales_df['Outlet_Size'])
big_mart_sales_df['Outlet_Location_Type'] = encoder.fit_transform(big_mart_sales_df['Outlet_Location_Type'])
big_mart_sales_df['Outlet_Type'] = encoder.fit_transform(big_mart_sales_df['Outlet_Type'])

print(big_mart_sales_df)


# Creating Feature Matrix (Independent Variables) & Target Variable (Dependent Variable)
# separating the data and labels
X = big_mart_sales_df.drop(columns = ['Item_Outlet_Sales'], axis=1) # Feature matrix
Y = big_mart_sales_df['Item_Outlet_Sales'] # Target variable

print(X)
print('-'*100)
print(Y)
print('-'*100)



# Data Standardization
scaler = StandardScaler()
print(scaler.fit(X))
print('-'*100)
standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data



# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=45)
print(X.shape, X_train.shape, X_test.shape)

# For Model Building
models = [LinearRegression, Lasso, Ridge, SVR, DecisionTreeRegressor, RandomForestRegressor]
mae_scores = []
mse_scores = []
rmse_scores = []
r2_scores = []

for model in models:
    regressor = model().fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))
    rmse_scores.append(mean_squared_error(y_test, y_pred, squared=False))
    r2_scores.append(r2_score(y_test, y_pred))

regression_metrics_df = pd.DataFrame({
    "Model": ["Linear Regression", "Lasso", "Ridge", "SVR", "Decision Tree Regressor", "Random Forest Regressor"],
    "Mean Absolute Error": mae_scores,
    "Mean Squared Error": mse_scores,
    "Root Mean Squared Error": rmse_scores,
    "R-squared (R2)": r2_scores
})

regression_metrics_df.set_index('Model', inplace=True)
print(regression_metrics_df)


