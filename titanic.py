#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv('titanic/train.csv')
df.head() #項目の確認

#%%
df.describe() #統計量の確認

#%%
df.hist(figsize= (12, 12))
#データチェック
#%%
plt.figure(figsize= (15,15))
sns.heatmap(df.corr(), annot = True)
#データチェック
#%%
sns.countplot('Sex', hue = 'Survived', data = df)
#データチェック

#%%
df.isnull().sum()
#nullチェック(前処理)

#%%
from sklearn.model_selection import train_test_split

#%%
#欠損値処理
df['Cabin'] = df['Cabin'].fillna("C85")
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna('S')

#%%
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
df['Embarked'] = df['Embarked'].map( {'S': 0 , 'C':1 , 'Q':2}).astype(int)
df = df.drop(['Cabin','Name','PassengerId','Ticket'],axis =1)
train_X = df.drop('Survived',axis = 1)
train_y = df.Survived
(train_X , test_X , train_y , test_y) = train_test_split(train_X, train_y , test_size = 0.3 , random_state = 0)

#%%
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state = 0)
clf = clf.fit(train_X , train_y)
pred = clf.predict(test_X)

#正解率の算出
from sklearn.metrics import (roc_curve , auc ,accuracy_score)
pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y , pred,pos_label = 1)
auc(fpr,tpr)
accuracy_score(pred,test_y)

#%%
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 10,max_depth=5,random_state = 0)
clf = clf.fit(train_X , train_y)
pred = clf.predict(test_X)
fpr, tpr , thresholds = roc_curve(test_y,pred,pos_label = 1)
auc(fpr,tpr)
accuracy_score(pred,test_y)

#%%
