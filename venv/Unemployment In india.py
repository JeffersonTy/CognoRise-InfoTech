import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime as dt
import calendar

import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("DataSet/Unemployment in india/Unemployment in India.csv")
print(data.head(10))
print('--'*100)

print(data.shape)

# Basic dataset information
print('Basic dataset information: ')
print(data.info)
print('--'*100)

# Descriptive statistics of the dataset
print('Descriptive statistics of the dataset: ')
print(data.describe())
print('--'*100)

# Check for missing values
print('Check for missing values: ')
print(data.isnull().sum())
print('--'*100)


# Check for possible duplicates
print('Check for possible duplicates: ')
print(data.duplicated().any())
print('--'*100)

# Value counts for every region
print('Value counts for every region: ')
print(data.Region.value_counts())
print('--'*100)


#updating column names
data.columns=['Region','date','frequency','estimated unemployment rate','estimated employed','estimated labour rate','area']

# Changing the data type of date attribute
data['date']=pd.to_datetime(data['date'],dayfirst=True)

# Extracting month from the date attribute
data['month_int']=data['date'].dt.month
print(data.head())
print('--'*100)

# Set plotting style
numeric_data = data.select_dtypes(include='number')

plt.style.use("ggplot")

plt.figure(figsize=(10, 10))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Histogram of Estimated Employment Rate by Area
plt.style.use("dark_background")

plt.figure(figsize=(10, 6))
sns.histplot(x="estimated employed", hue="area", data=data, kde=True, palette="Set2")
plt.title("Histogram of Estimated Employment Rate by Area")
plt.xlabel("Estimated Employment Rate")
plt.ylabel("Count")
plt.show()

# Histogram of Estimated Unemployment Rate by Area
plt.style.use("dark_background")

plt.figure(figsize=(10, 6))
sns.histplot(x="estimated unemployment rate", hue="area", data=data, kde=True, palette="Set2")
plt.title("Histogram of Estimated Unemployment Rate by Area")
plt.xlabel("Estimated Unemployment Rate")
plt.ylabel("Count")
plt.show()

# BoxPlot of Estimated UnEmployement Rate by Region
data = data[['Region', 'estimated unemployment rate']]
sns.boxplot(x='Region', y='estimated unemployment rate', data=data)
plt.xlabel('Region')
plt.ylabel('Estimated Unemployment Rate')
plt.title('Box Plot of Estimated Unemployment Rate by Region')
plt.xticks(rotation=90)
plt.show()

# Mean Estimated UnEmployement Rate by Region
data = data[['Region', 'estimated unemployment rate']]
data_grouped = data.groupby('Region')['estimated unemployment rate'].mean()
data_grouped.plot(kind='bar')
plt.xlabel('Region')
plt.ylabel('Mean Estimated Unemployment Rate')
plt.title('Mean Estimated Unemployment Rate by Region')
plt.show()

# Distribution of Estimated UnEmployement Rate by Region
data = data[['Region', 'estimated unemployment rate']]
data_grouped = data.groupby('Region')['estimated unemployment rate'].sum()
data_grouped.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Estimated Unemployment Rate by Region')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


