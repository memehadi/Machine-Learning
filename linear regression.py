#!/usr/bin/env python
# coding: utf-8

# In[7]:



import os
import numpy as np
import pandas as pd
from pandas import read_csv
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv('Placement.csv')
print (data.head());
data = data[["degree_p", "mba_p","status"]]
#coverting strings into integers
le=preprocessing.LabelEncoder()
data.status=le.fit_transform(list(data["status"]))
predict = "status"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)




# In[ ]:




