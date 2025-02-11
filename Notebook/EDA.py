import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
wasrnings.filterwarnings('ignore')

### Import Dataset
df = pd.read_csv("D:/ML Project/Notebook/PM.csv")

###EDA
df.head()
df.shape

#check missing value
df.isna().sum()

#check duplicates
df.duplicated().sum()

#check null and dtype
df.info()

#checking the numbers of unique values in each column
df.nunique()

#check the statistics of dataset
df.describe()

#Exploring the data
#print("Categories in 'Product ID' varible:     ",end=" ")
#print(df['Product ID'].unique())

print("Categories in 'Type' varible:     ",end=" ")
print(df['Type'].unique())

print("Categories in 'Air temperature [K]' varible:     ",end=" ")
print(df['Air temperature [K]'].unique())

print("Categories in 'Process temperature [K]' varible:     ",end=" ")
print(df['Process temperature [K]'].unique())

print("Categories in 'Rotational speed [rpm]' varible:     ",end=" ")
print(df['Rotational speed [rpm]'].unique())

print("Categories in 'Torque [Nm]' varible:     ",end=" ")
print(df['Rotational speed [rpm]'].unique())

print("Categories in 'Tool wear [min]' varible:     ",end=" ")
print(df['Tool wear [min]'].unique())

##Feature Engineering
#Define Numerical and Caegorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'o']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'o']

print('we have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))