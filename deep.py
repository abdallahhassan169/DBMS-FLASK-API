# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 01:44:33 2021

@author: abdal
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('diabetes.csv')



drops=['Pregnancies','SkinThickness']
df.drop(drops,inplace=True,axis=1)

df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())


X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


import tensorflow.python as tf

learning_schedule = tf.keras.optimizers.ExponentialDecay(
initial_learning_rate = 1e-5,
decay_steps = .02,
decay_rate = 1-(1e-5))


# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=50, activation='sigmoid'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=50, activation='sigmoid'))

ann.add(tf.keras.layers.Dense(units=50, activation='sigmoid'))


ann.add(tf.keras.layers.Dense(units=50, activation='sigmoid'))



# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='relu'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 100, epochs = 100)

# Part 4 - Making the predictions and evaluating the model


# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
