# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Importing the Adataset
Adataset = pd.read_csv('Accident_Dataset.csv')
X1 = Adataset.iloc[:,:].values
X1 = np.delete(X1,2,axis=1)
X1 = np.delete(X1,1,axis=1)
X1 = np.delete(X1,0,axis=1)
names = (Adataset.columns.values)
names = names[3:]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy="most_frequent",axis=0)
imputer = imputer.fit(X1[:,:])
X1[:,:] =  imputer.transform(X1[:,:])
data_new = pd.DataFrame(data=X1,columns=names)

imputer = Imputer(missing_values=-1,strategy="most_frequent",axis=0)
imputer = imputer.fit(X1[:,:])
X1[:,:] =  imputer.transform(X1[:,:])
data_new = pd.DataFrame(data=X1,columns=names)

imputer = Imputer(missing_values=-2,strategy="most_frequent",axis=0)
imputer = imputer.fit(X1[:,:])
X1[:,:] =  imputer.transform(X1[:,:])
data_new = pd.DataFrame(data=X1,columns=names)

imputer = Imputer(missing_values=1,strategy="most_frequent",axis=0)
imputer = imputer.fit(X1[:,:])
X1[:,:] =  imputer.transform(X1[:,:])
data_new = pd.DataFrame(data=X1,columns=names)


plt.figure(figsize=(30,8))

sns.countplot(x='LIGHT_CONDITION', hue='SEVERITY',data=data_new)
sns.countplot(x='SEVERITY',hue='ALCOHOLTIME',data=data_new)
sns.countplot(x='LIGHT_CONDITION',hue='ALCOHOLTIME',data=data_new)
sns.countplot(x='SEVERITY',hue='SPEED_ZONE',data=data_new)
sns.countplot(x='POLICE_ATTEND',hue='SPEED_ZONE',data=data_new)
sns.countplot(x='FEMALES',hue='SEVERITY',data=data_new)