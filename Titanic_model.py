#!/usr/bin/env python
# coding: utf-8

# # Import the library

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[70]:


train=pd.read_csv('datasets/train.csv')
test=pd.read_csv('datasets/test.csv')


# # EDA

# In[140]:


train.head()


# In[141]:


train.isna().sum()


# In[142]:


test.isna().sum()


# In[143]:


dataset=pd.concat((train, test), axis=0)


# In[144]:


dataset.set_index('PassengerId',inplace=True)


# In[145]:


dataset['Age'].plot.hist(bins=50);


# In[146]:


dataset['Age'].describe()


# In[147]:


age_mean=dataset['Age'].mean()
age_std=dataset['Age'].std()
size=dataset['Age'].isna().sum()
random_list=np.random.randint(low=age_mean-age_std, high=age_mean+age_std, size=size)
dataset['Age'][dataset['Age'].isna()]=random_list


# In[148]:


dataset['Age'].plot.hist(bins=50);


# In[149]:


dataset['Age'].describe()


# In[150]:


sns.countplot('Embarked', data=dataset);


# In[151]:


dataset['Embarked'].fillna('S', inplace=True)


# In[152]:


dataset.isna().sum()


# In[153]:


dataset['Family_size']=dataset['SibSp']+dataset['Parch']+1


# In[154]:


sns.countplot('Family_size', data=dataset)


# In[155]:


dataset['Fare'][dataset['Fare'].isna()]


# In[156]:


dataset.loc[1044]


# In[157]:


dataset.loc[((dataset['Pclass']==3) & (dataset['Family_size']==1)),'Fare'].plot.hist()


# In[158]:


dataset.loc[((dataset['Pclass']==3) & (dataset['Family_size']==1)),'Fare'].median()


# In[159]:


dataset.loc[((dataset['Pclass']==3) & (dataset['Family_size']==2)),'Fare'].plot.hist();


# In[160]:


dataset['Fare'].fillna(7.8542, inplace=True)


# In[161]:


dataset.drop('Cabin', axis=1, inplace=True)


# In[162]:


dataset.head()


# In[163]:


sns.countplot('Pclass', hue='Survived', data=dataset)


# In[164]:


sns.countplot('Embarked', hue='Survived', data=train)


# In[165]:


# pclass=(train.groupby('Pclass')['Survived'].mean()*10)/2.423625
# pclass


# In[166]:


train.groupby('Embarked')['Survived'].mean()


# In[167]:


sns.countplot('Embarked', hue='Pclass', data=test);


# In[168]:


# dataset['Pclass']=dataset['Pclass'].map(pclass) 


# In[169]:


dataset=pd.concat((dataset,pd.get_dummies(dataset.Embarked, drop_first=True,columns={'Q':'Queenstown','S':'Southampton'})), axis=1)


# In[ ]:





# In[170]:


dataset.groupby('Sex')["Survived"].mean()


# In[171]:


sex={'male':0,'female':1}
dataset['Sex']=dataset['Sex'].map(sex)


# In[172]:


dataset


# In[173]:


dataset.drop(['Name', 'SibSp', 'Parch', 'Ticket','Embarked'], axis=1,inplace=True)


# In[174]:


dataset.head()


# In[175]:


dataset.rename(columns={'Q':'Queenstown','S':'Southampton'},inplace=True)


# In[176]:


dataset


# In[177]:


x=dataset.drop('Survived', axis=1)
y=dataset['Survived'].values


# In[178]:


sns.distplot(x['Fare']);


# In[179]:


x['Fare']=x['Fare'].map(lambda i:np.log(i) if i>0 else 0)


# In[180]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x)
x=sc.transform(x)


# In[113]:


train_x=x[:891,:]
test_x=x[891:1310,:]
train_y=y[:891]


# # Model Building

# In[114]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[115]:


from sklearn.model_selection import KFold,cross_val_score
kfold=KFold(n_splits=10,shuffle=True, random_state=0)


# In[116]:


clf=LogisticRegression(max_iter=1000)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[117]:


score.mean()


# In[118]:


clf=KNeighborsClassifier(n_neighbors=12)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[119]:


score.mean()


# In[120]:


clf=SVC(random_state=0)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[121]:


score.mean()


# In[122]:


clf=GaussianNB()
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[123]:


score.mean()


# In[124]:


clf=DecisionTreeClassifier(random_state=0)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[125]:


score.mean()


# In[126]:


clf=RandomForestClassifier(n_estimators=12, random_state=0)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[127]:


score.mean()


# In[128]:


clf=SVC(random_state=0)
clf.fit(train_x,train_y)


# In[129]:


y_pred=clf.predict(test_x).astype(int)


# In[130]:


PassengerId=pd.read_csv('datasets/gender_submission.csv')['PassengerId']
pd.DataFrame({'Survived':y_pred}, index=PassengerId).to_csv('titanic_pred.csv')


# ### Dumping the model and preprocessing scale

# In[131]:


import pickle
pickle_out=open('classifier.pkl','wb')
pickle.dump(clf,pickle_out)
pickle_out.close()


# In[132]:


import pickle
pickle_out=open('scale.pkl','wb')
pickle.dump(sc,pickle_out)
pickle_out.close()


# In[ ]:




