import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv('DataSet/Employee salaries for different job roles/ds_salaries.csv')

print("First few rows of the dataset:")
print(df.head())
print('-'*100)

# Check for missing values in each column
print(df.isnull().sum())
print('-'*100)

# Get summary statistics for numerical columns
summary_stats = df.describe()
print("\nSummary statistics for numerical columns:")
print(summary_stats)
print('-'*100)



# EDA AND DATA VIZUALIZATION
def scatterplot(x:str, y:str, title:str):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, data=df,hue=x, legend=False, palette="rainbow")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def barplot(x:str, y:str, title:str, rot=None):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=x, y=y, data=df, hue=x, legend=False, palette='rainbow')
    plt.xticks(rotation=rot)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

# Salary Distribution Across Job Titles
barplot(x='job_title', y='salary', title='Salary Distribution Across Job Titles', rot=90)

# Salary Distribution Across Experience Levels
barplot(x='experience_level', y='salary', title='Salary Distribution Across Experience Levels')

# Salary Distribution Across Employment Types
barplot(x='employment_type', y='salary', title='Salary Distribution Across Employment Types')

# Visualize relationships using scatter plots and histograms
scatterplot(x='remote_ratio', y='salary', title='Relationship between Remote Ratio and Salary')

# Relationship between Company Size and Salary
scatterplot(x='company_size', y='salary', title='Relationship between Company Size and Salary')

# Visualize trends in remote work percentages and company sizes
barplot(x='remote_ratio', y='salary', title='Impact of Remote Ratio on Salary')

# Impact of Company Size on Salary
barplot(x='company_size', y='salary', title='Impact of Company Size on Salary')

# Feature Engineering: Calculate average salary per job title
avg_salary_per_title = df.groupby('job_title')['salary'].mean().reset_index()
avg_salary_per_title.rename(columns={'salary': 'avg_salary_per_title'}, inplace=True)
df = df.merge(avg_salary_per_title, on='job_title', how='left')

# Feature Engineering: Calculate average salary per experience level
avg_salary_per_exp = df.groupby('experience_level')['salary'].mean().reset_index()
avg_salary_per_exp.rename(columns={'salary': 'avg_salary_per_experience'}, inplace=True)
df = df.merge(avg_salary_per_exp, on='experience_level', how='left')

# Predictive Analysis (Optional):
X = df[['avg_salary_per_title', 'avg_salary_per_experience']]
y = df['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select a regression model and train it on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict salaries on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Create a scatter plot to visualize original vs. predicted salaries
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.xlabel('Original Salary')
plt.ylabel('Predicted Salary')
plt.title('Original vs. Predicted Salary')
plt.show()

# Plot a histogram of the residuals (difference between original and predicted salaries)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

