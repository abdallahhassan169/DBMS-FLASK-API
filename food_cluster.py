# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:54:58 2021

@author: abdal
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('cluster_weight.csv')


drops=['id','hdl','ratio','stab_glu','location','frame','weight_1','bp_1d','bp_2d','bp_1s','bp_2s','time_ppn']
df.drop(drops,inplace=True,axis=1)

df.fillna(value=df.mean(),inplace=True)

X=df.values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

arr=y_kmeans
df['cluster'] = arr.tolist()

X2 = df.iloc[:, 0:-1].values
y2 = df.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X2 = np.array(ct.fit_transform(X2))
X2=X2[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
arr=[[228.0,4.640000,58,0,61.0,243,49.0,57.0]]
arr=np.array(arr)
xv=classifier.predict(arr)
from joblib import dump

# dump the pipeline model
dump(classifier, filename="food_cluster.joblib")
