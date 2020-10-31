#!/usr/bin/env python
# coding: utf-8

# # Import the library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


train=pd.read_csv('datasets/train.csv')
test=pd.read_csv('datasets/test.csv')


# # EDA

# In[3]:


train.head()


# In[4]:


train.isna().sum()


# In[5]:


test.isna().sum()


# In[6]:


dataset=pd.concat((train, test), axis=0)


# In[7]:


dataset.set_index('PassengerId',inplace=True)


# In[8]:


dataset['Age'].plot.hist(bins=50);


# In[9]:


dataset['Age'].describe()


# In[10]:


age_mean=dataset['Age'].mean()
age_std=dataset['Age'].std()
size=dataset['Age'].isna().sum()
random_list=np.random.randint(low=age_mean-age_std, high=age_mean+age_std, size=size)
dataset['Age'][dataset['Age'].isna()]=random_list


# In[11]:


dataset['Age'].plot.hist(bins=50);


# In[12]:


dataset['Age'].describe()


# In[13]:


sns.countplot('Embarked', data=dataset);


# In[14]:


dataset['Embarked'].fillna('S', inplace=True)


# In[15]:


dataset.isna().sum()


# In[16]:


dataset['Family_size']=dataset['SibSp']+dataset['Parch']+1


# In[17]:


sns.countplot('Family_size', data=dataset)


# In[18]:


dataset['Fare'][dataset['Fare'].isna()]


# In[19]:


dataset.loc[1044]


# In[20]:


dataset.loc[((dataset['Pclass']==3) & (dataset['Family_size']==1)),'Fare'].plot.hist()


# In[21]:


dataset.loc[((dataset['Pclass']==3) & (dataset['Family_size']==1)),'Fare'].median()


# In[22]:


dataset.loc[((dataset['Pclass']==3) & (dataset['Family_size']==2)),'Fare'].plot.hist();


# In[23]:


dataset['Fare'].fillna(7.8542, inplace=True)


# In[24]:


dataset.drop('Cabin', axis=1, inplace=True)


# In[25]:


dataset.head()


# In[26]:


sns.countplot('Pclass', hue='Survived', data=dataset)


# In[27]:


sns.countplot('Embarked', hue='Survived', data=train)


# In[28]:


# pclass=(train.groupby('Pclass')['Survived'].mean()*10)/2.423625
# pclass


# In[29]:


train.groupby('Embarked')['Survived'].mean()


# In[30]:


sns.countplot('Embarked', hue='Pclass', data=test);


# In[31]:


# dataset['Pclass']=dataset['Pclass'].map(pclass) 


# In[32]:


dataset=pd.concat((dataset,pd.get_dummies(dataset.Embarked, drop_first=True,columns={'Q':'Queenstown','S':'Southampton'})), axis=1)


# In[ ]:





# In[33]:


dataset.groupby('Sex')["Survived"].mean()


# In[34]:


sex={'male':0,'female':1}
dataset['Sex']=dataset['Sex'].map(sex)


# In[35]:


dataset


# In[36]:


dataset.drop(['Name', 'SibSp', 'Parch', 'Ticket','Embarked'], axis=1,inplace=True)


# In[37]:


dataset.head()


# In[38]:


dataset.rename(columns={'Q':'Queenstown','S':'Southampton'},inplace=True)


# In[39]:


dataset


# In[40]:


x=dataset.drop('Survived', axis=1)
y=dataset['Survived'].values


# In[41]:


sns.distplot(x['Fare']);


# In[42]:


x['Fare']=x['Fare'].map(lambda i:np.log(i) if i>0 else 0)


# In[43]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


# In[48]:


train_x=x[:891,:]
test_x=x[891:1310,:]
train_y=y[:891]


# # Model Building

# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[50]:


from sklearn.model_selection import KFold,cross_val_score
kfold=KFold(n_splits=10,shuffle=True, random_state=0)


# In[51]:


clf=LogisticRegression(max_iter=1000)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[52]:


score.mean()


# In[53]:


clf=KNeighborsClassifier(n_neighbors=12)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[54]:


score.mean()


# In[55]:


clf=SVC(random_state=0)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[56]:


score.mean()


# In[57]:


clf=GaussianNB()
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[58]:


score.mean()


# In[59]:


clf=DecisionTreeClassifier(random_state=0)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[60]:


score.mean()


# In[61]:


clf=RandomForestClassifier(n_estimators=12, random_state=0)
score=cross_val_score(clf, train_x,train_y, cv=kfold)
score


# In[62]:


score.mean()


# In[63]:


clf=SVC(random_state=0)
clf.fit(train_x,train_y)


# In[64]:


y_pred=clf.predict(test_x).astype(int)


# In[65]:


PassengerId=pd.read_csv('datasets/gender_submission.csv')['PassengerId']
pd.DataFrame({'Survived':y_pred}, index=PassengerId).to_csv('titanic_pred.csv')


# ### Dumping the model and preprocessing scale

# In[66]:


import pickle
pickle_out=open('classifier.pkl','wb')
pickle.dump(clf,pickle_out)
pickle_out.close()


# In[68]:


import pickle
pickle_out=open('scale.pkl','wb')
pickle.dump(sc,pickle_out)
pickle_out.close()


# In[ ]:




