#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 

df = pd.read_csv("movies.csv")
dict = {}
x = 0
for i in range(len(df)):
    arr = df.genres[i].split('|')
    for j in arr:
        if (j not in dict):
            dict[j] = x
            x+=1
            
print (dict)

for i in dict:
    if (i!='(no genres listed)'):
        df[i] = [0]*len(df)

for i in range(len(df)):
    arr = df.genres[i].split('|')
    for j in arr:
        if (j!='(no genres listed)'):
            df[j][i] = 1
df = df.drop(['genres'],1)

df = df.drop(['title'],1)




# In[25]:


X = df
sc = StandardScaler()
X = sc.fit_transform(X) 

 


# In[26]:


K = range(1,30)
Sum_of_squared_distances = []
for k in K:
    km = KMeans(n_clusters = k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('No of Clusters')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method for Optimal k')
plt.show()


# In[31]:


model = KMeans(n_clusters=14).fit(X)



# Determine the cluster labels of new_points: labels
df['cluster'] = model.predict(X)
print (df.head)

pca = PCA(n_components = 2) 
X = pca.fit_transform(X)

plt.scatter (X[ : , 0], X[ : , 1],  c= model.labels_, s=50, alpha=0.7)
plt.show()


# In[28]:


ratings = pd.read_csv("movies.csv")

