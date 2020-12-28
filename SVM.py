#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from pandas import read_csv
import sklearn
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
data = pd.read_csv('Placement.csv')
#print (data.head());
data = data[["degree_p", "mba_p","status"]]
#coverting strings into integers
le=preprocessing.LabelEncoder()
data.status=le.fit_transform(list(data["status"]))

predict = "status"
x = list(zip(data.degree_p,data.mba_p))
y = list(data.status)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

clf=svm.SVC(kernel="poly", degree=3)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
acc=metrics.accuracy_score(y_test,y_pred)
print(acc)
 


# In[ ]:





# In[ ]:




