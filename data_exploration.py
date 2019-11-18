#%% Import the required libraries

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


#%% Load the data and show some statistics
data = pd.read_csv('HAM10000_metadata.csv')
print(data.head())

print(data.isnull().sum())

#%% Previous command shows us that age has some missing data

null = data.age.isnull().values
print(type(null))
result = np.where(null == True)
print(len(result))

# we found the values where age is null.
# Let's impute the nan values
y = data['dx']
X = np.array(data.drop(['dx'], 1))
age = np.array(data['age'])
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_norm = imp_mean.fit_transform(X)
# print(age_norm.shape)
# print(y.shape)

# encoder = OneHotEncoder(handle_unknown='ignore')
# X_encoded = encoder.fit_transform(X)
# print(X_encoded[0:5])
#
# y = data['dx']

#%% PCA
Ncomponents=2
pca = PCA()

X_reduced = pca.fit_transform(X)

print(X_reduced.shape)