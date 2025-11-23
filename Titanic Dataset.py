import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv(r"c:\Users\Sheeza\Downloads\Titanic-Dataset.csv")
df.info()
df.describe()
print(df)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])
df['Embarked']=le.fit_transform(df['Embarked'])
X=df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    min_samples_leaf=2,
    min_samples_split=4,
    class_weight='balanced'
)
    
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Model Accuracy:{accuracy*100:.2f}%")
cm=confusion_matrix(y_test,y_pred)
print(cm)
