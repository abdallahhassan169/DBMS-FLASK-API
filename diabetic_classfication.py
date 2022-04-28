import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('diabetes.csv')



drops=['Pregnancies','SkinThickness','DiabetesPedigreeFunction']
df.drop(drops,inplace=True,axis=1)

df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
mean=X_train.mean()
print(mean)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0).fit(X_train, y_train)



pred = classifier.predict(X_test)


arr=[[85.0	,66.0	,79.799479,	26.6,	31]]
arr=np.array(arr)
test=classifier.predict(arr)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred, y_test)
print(cm)



from joblib import dump

# dump the pipeline model
dump(classifier, filename="diabetic_classfication.joblib")
