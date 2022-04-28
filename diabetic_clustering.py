import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df_full=pd.read_csv('diabetes.csv')
df1=df_full[df_full['Outcome']==0].sample(n=268)
df2=df_full[df_full['Outcome']==1].sample(n=268)
df=pd.concat([df1,df2])


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
mean=X_train.mean()
print(mean)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0).fit(X_train, y_train)

y_pred=classifier.predict(X_test)


mean=y_test.mean()


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from joblib import dump

# dump the pipeline model
dump(classifier, filename="diabetic_classfication.joblib")
