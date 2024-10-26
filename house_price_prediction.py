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

# ## Import Libraries

import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import pandas as pd 

# # Importing Data

# / reads the csv file and imports to dataframe object 'data'
data = pd.read_csv(r'C:\Users\44734\OneDrive\Desktop\Python Practice\Data Sets\housing.csv')

# # Data Modification / Cleaning

# / drops all NaN values from the (df)
data.dropna(inplace=True)

# / checks if all NaN / null values are dropped 
data.info()

# / imports the test train split class
from sklearn.model_selection import train_test_split

# / assigns the training data into two separate (df)s
X = data.drop(['median_house_value'], axis=1) # X is all columns but 'median house value'
y = data['median_house_value'] # y is only median house value

# / separates into four different datasets using train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # test size = 25% of original data

# / froms training data by joining X_train and y_train
train_data = X_train.join(y_train)

# / displays train_data to check if all is correct
train_data

# # Visualising Data

# / displays histograms of our data, dropping 'ocean proximity' as it is string values
train_data.drop(['ocean_proximity'],axis = 1).hist(figsize=(15, 8))

# / 
plt.figure(figsize= (15, 8))
sns.heatmap(train_data.drop(['ocean_proximity'],axis = 1).corr(), annot = True, cmap='YlGnBu')

# # Pre-processing

# / transforming each feature to its natural logarithm to normalize the distribution and stabalize variance
train_data['total_rooms'] = np.log(train_data['total_rooms']+1) # each feature is transformed, being pulled from the train_data dataframe, 1 is added to each item to avoid taking the (undefined) natural logarithm of 0
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms']+1)
train_data['population'] = np.log(train_data['population']+1)
train_data['households'] = np.log(train_data['households']+1)

# / plots new histograms (without the ocean proximity string data) using natural logarithms 
train_data.drop(['ocean_proximity'],axis = 1).hist(figsize=(15, 8)) 

# // this converts our catagorical data into binary (true / false), and joins it to the training data and also drops the original 
# / string data from the training data while permantly changing the train_data
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity, dtype=int)).drop(['ocean_proximity'], axis = 1)

# / creates a correlation heatmap between our variabales
plt.figure(figsize= (15, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')

# / visualises the median house prices on a map (not scaled), where colour/hue is the median house value
plt.figure(figsize=(15,8))
sns.scatterplot(x='latitude', y='longitude', data = train_data, hue='median_house_value', palette='coolwarm')

# # Feature Engineering

# / creates a new column that includes our new feature 
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

# # Linear Regression Model

# / import model package
from sklearn.linear_model import LinearRegression

# +
# / splits our data into training target and input data
X_train, y_train = train_data.drop(['median_house_value'], axis = 1), train_data['median_house_value']

# / define the type of regression into 'reg' variable
reg = LinearRegression()

# / fits the data to the model
reg.fit(X_train, y_train)

# +
# / create a new test data variable
test_data = X_test.join(y_test)

# / transforms training data into its natural logarithms 
test_data['total_rooms'] = np.log(test_data['total_rooms']+1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms']+1)
test_data['population'] = np.log(test_data['population']+1)
test_data['households'] = np.log(test_data['households']+1)

# / converts the string data [ocean_proximity] in our test_data to binary catagorical values
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity, dtype=int)).drop(['ocean_proximity'], axis = 1)

# / creates the new features for test_data as we previously did for training_data 
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']
# -

# / creates target and input test data using the transformed features 
X_test, y_test = test_data.drop(['median_house_value'], axis = 1), test_data['median_house_value']

# ## Model Evaulation

# // generates a r^2 score to test how well the model fits our data
# / where it incdicates the proportion of the variance in the dependent variable that is predictable from the independent variables
reg.score(X_test, y_test)

# ## Model Visualisation

# / create prediction data
lr_line = reg.predict(X_train)

# / create plot of highest correlation variable
sns.lmplot(x='median_income', y='median_house_value', data=data, scatter_kws={'alpha':0.1})
plt.ylim(0, 500000)
plt.xlabel('median income')
plt.ylabel('median house value')
plt.title('median income & median house value regression')
plt.show


