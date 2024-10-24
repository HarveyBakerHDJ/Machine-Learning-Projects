# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### Goal

# +
# 1 - analyse the correlation of X variables with the y output (purchase amount)
# 2 - implement simple linear regression model with multiple variables  

# +
# this model will use the 'customer purhcase behaviour' dataset from Kaggle
# -

# ### Importing Modules

# %matplotlib inline

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ### Importing Data

# / import data to a pandas dataframe
df = pd.read_csv(r'C:\Users\44734\OneDrive\Desktop\Python Practice\Data Sets\cpb.csv') # pulls data from directory
df # displays dataframe

# ## Data Preparation

df.info # provides a summary of the dataframe

# / remove empty data
df.dropna(inplace=True) # removes all NaN values

# / remove and store user ids
user_ids = df.drop(['age','annual_income','purchase_amount','loyalty_score','region','purchase_frequency'], axis = 1) # creates new dataframe for our user_ids 
df = df.drop(['user_id'], axis = 1) # creates a new dataframe without our user_ids column

# / remove the discrete / string values from the dataframe
df = df.drop(['region'], axis =1) # removes string format data

# #### Correlation Analysis

# displays a matrix with correlation values between X variables 
df.corr()

# +
# the correlation matrix explains that the highest variable correlated to 'purchase amount' is 'loyalty score', closely followed by all 3 other 
# variables
# -

# #### Test / Train Split

# import module
from sklearn.model_selection import train_test_split

# / create separate X_train and y_train variables
# assigns the training data into two separate (df)s
X1 = df.drop(['purchase_amount'], axis=1) # all input variables
y1 = df['purchase_amount'] # target variable

# / splits into testing & training for both X and y 
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25) # 25% of dataset is testing data

# / ensures they only contain correct input or target variables
X1_train, y1_train = df.drop(['purchase_amount'], axis = 1), df['purchase_amount']
X1_train, y1_train

# ## Simple Linear Regression

# / sklearn model
from sklearn.linear_model import LinearRegression # imports the linear regression class

# / define the model within a variable
model1 = LinearRegression() # assigns it to 'model1'

# #### Fitting the Model

model1.fit(X1_train, y1_train) # fits linear regression model to our training data

LR_line = model1.predict(X1_train) # creates a regression line for later use in our plots, predicts based on target variable

# #### Visualise the Models

# +
# / chart blocks
# loyalty score and purchase amount
sns.lmplot(x="loyalty_score", y="purchase_amount", data=df) 
plt.xlabel("Loyalty Score")
plt.ylabel("Purchase Amount")
plt.title("Linear Regression: X vs. y")
plt.show()

# age and purchase amount
sns.lmplot(x="age", y="purchase_amount", data=df) 
plt.xlabel("Age")
plt.ylabel("Purchase Amount")
plt.title("Linear Regression: X vs. y")
plt.show()

# purchase frequency
sns.lmplot(x="purchase_frequency", y="purchase_amount", data=df) 
plt.xlabel("Purchase Frequency")
plt.ylabel("Purchase Amount")
plt.title("Linear Regression: X vs. y")
plt.show()

# annual income and purchase amount
sns.lmplot(x="annual_income", y="purchase_amount", data=df) 
plt.xlabel("Annual Income")
plt.ylabel("Purchase Amount")
plt.title("Linear Regression: X vs. y")
plt.show()
