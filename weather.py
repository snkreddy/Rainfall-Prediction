
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold


# In[2]:


# Read the train and test data
train=pd.read_csv("weatherAUS.csv")
# test=pd.read_csv("test_2nAIblo.csv")


# In[3]:


train.columns


# In[4]:


train.dtypes


# In[5]:


train.head()


# In[6]:


train.shape


# In[7]:


train.nunique()


# In[8]:


# Univariate Analysis


# In[9]:


train.RainTomorrow.value_counts(normalize=True)


# In[10]:


# Around 77% trainee have no rain


# In[11]:


train['Location'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Location')
plt.show()


train['WindGustDir'].value_counts(normalize=True).plot.bar(title= 'WindGustDir')
plt.show()

train['WindDir9am'].value_counts(normalize=True).plot.bar(title= 'WindDir9am')
plt.show()


train['WindDir3pm'].value_counts(normalize=True).plot.bar(title= 'WindDir3pm')
plt.show()


# In[12]:


df=train.dropna()


# In[13]:


sns.distplot(df['Rainfall'])


# In[14]:


sns.distplot(df['Sunshine'])


# In[15]:


sns.distplot(df['MaxTemp'])


# In[16]:


sns.distplot(df['MinTemp'])


# In[17]:


sns.distplot(df['Evaporation'])


# In[18]:


sns.distplot(df['WindGustSpeed'])


# In[19]:


sns.distplot(df['Humidity3pm'])


# In[20]:


# Bivariate Analysis
matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# In[21]:


# Much corellation between the data


# In[22]:


# Mutivariant analysis 


# In[23]:


sns.boxplot(x="Cloud9am", y="MaxTemp", hue="RainTomorrow", data=train ,linewidth=1)


# In[45]:


sns.boxplot(x="RainTomorrow", y="MaxTemp", data=train ,linewidth=1)


# In[46]:


sns.boxplot(x="RainTomorrow", y="MaxTemp", data=train ,linewidth=1)


# In[47]:


sns.boxplot(x="RainTomorrow", y="MinTemp", data=train ,linewidth=1)


# In[48]:


sns.boxplot(x="RainTomorrow", y="Sunshine", data=train ,linewidth=1)


# In[49]:


sns.boxplot(x="RainTomorrow", y="WindGustSpeed", data=train ,linewidth=1)


# In[53]:


sns.boxplot(x="RainTomorrow", y="Evaporation", data=train ,linewidth=.5)
# Missing Values Treatment


# In[55]:


sns.boxplot(x="RainTomorrow", y="Humidity3pm", data=train ,linewidth=.5)


# In[56]:


sns.boxplot(x="RainTomorrow", y="Humidity9am", data=train ,linewidth=.5)


# In[63]:


sns.set(rc={'figure.figsize':(30,8)})
sns.barplot(x="Location", y="RainTomorrow", data=train ,linewidth=.5)


# In[66]:


sns.set(rc={'figure.figsize':(5,5)})
sns.boxplot(y="Temp9am", x="RainTomorrow", data=train ,linewidth=.5)


# In[67]:


sns.boxplot(y="Temp3pm", x="RainTomorrow", data=train ,linewidth=.5)


# In[68]:


sns.boxplot(x="RainToday", y="MaxTemp", hue="RainTomorrow", data=train ,linewidth=1)


# In[69]:


sns.barplot(x="RainToday", y="RainTomorrow", data=train ,linewidth=.5)


# In[70]:


sns.boxplot(x="RainToday", y="Pressure9am", data=train ,linewidth=.5)


# In[71]:


sns.boxplot(x="RainToday", y="Pressure3pm", data=train ,linewidth=.5)


# In[72]:


sns.boxplot(x="RainToday", y="Cloud9am", data=train ,linewidth=.5)


# In[73]:


sns.boxplot(x="RainToday", y="Cloud3pm", data=train ,linewidth=.5)


# In[25]:



# Check the number of missing values in each variable
train.isnull().sum()


# In[26]:


# except location and date rest all the columns have missing values


# In[27]:


df=train.copy()


# In[28]:


def f(x):
    if(len(str(x))==3):
        
        return 1
    else:
        return 0
df.RainTomorrow=df.RainTomorrow.apply(lambda x :f(x) )
df.RainToday=df.RainToday.apply(lambda x :f(x) )
df.Date=df.Date.apply(lambda x :str(x).replace("-"," ")[4:])


# In[29]:


df=df.assign(month=df.Date.apply(lambda x :int(str(x)[0:3])))
df=df.assign(ID=np.arange(len(df)))
df.head()
# df.assign(date=df.Date.apply(lambda x :str(x)[0:3]))
# del df['Date']


# In[30]:


del df['Date']
df=df.dropna(subset=['RainToday','RainTomorrow'])


# In[32]:


df[['MinTemp', 'MaxTemp','Rainfall','Evaporation','Sunshine','WindSpeed9am','WindSpeed3pm','WindGustSpeed']] = df.groupby(['Location', 'month','RainToday'])['MinTemp', 'MaxTemp','Rainfall','Evaporation','Sunshine','WindSpeed9am','WindSpeed3pm','WindGustSpeed'].transform(lambda x: x.fillna(x.median()))
df[['Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']] = df.groupby(['Location', 'month','RainToday'])['Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']    .transform(lambda x: x.fillna(x.median()))
    
df.isnull().sum()


# In[33]:


df[['MinTemp', 'MaxTemp','Rainfall','Evaporation','Sunshine','WindSpeed9am','WindSpeed3pm','WindGustSpeed']] = df.groupby(['Location', 'RainToday'])['MinTemp', 'MaxTemp','Rainfall','Evaporation','Sunshine','WindSpeed9am','WindSpeed3pm','WindGustSpeed'].transform(lambda x: x.fillna(x.median()))

df[['Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']] = df.groupby(['Location', 'RainToday'])['Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']    .transform(lambda x: x.fillna(x.median()))

df.isnull().sum()


# In[34]:


df[['MinTemp', 'MaxTemp','Rainfall','Evaporation','Sunshine','WindSpeed9am','WindSpeed3pm','WindGustSpeed']] = df.groupby(['RainToday', 'month'])['MinTemp', 'MaxTemp','Rainfall','Evaporation','Sunshine','WindSpeed9am','WindSpeed3pm','WindGustSpeed'].transform(lambda x: x.fillna(x.median()))

df[['Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']] = df.groupby(['RainToday', 'month'])['Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']    .transform(lambda x: x.fillna(x.median()))

df.isnull().sum()


# In[35]:


# categorical data 
df['WindGustDir'].fillna(df['WindGustDir'].mode()[0], inplace=True)
df['WindDir9am'].fillna(df['WindDir9am'].mode()[0], inplace=True)
df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0], inplace=True)
df.isnull().sum()


# In[36]:


del df['RISK_MM']


# In[37]:



train = df 

test = df.sample(8000)

train=train[train.ID.isin(test.ID) == False]

del train['ID']
del test['ID']


# In[38]:


# train=train.apply(preprocessing.LabelEncoder().fit_transform)
y = (train.RainTomorrow)
X = train.drop('RainTomorrow',1)
y1=(test.RainTomorrow)
test=test.drop('RainTomorrow',1)


# In[39]:


# print(train.head(20))


# In[40]:


X=X.apply(preprocessing.LabelEncoder().fit_transform)
test=test.apply(preprocessing.LabelEncoder().fit_transform)


# In[42]:


# print(np.isfinite(X).all())
# Logistic regression using 10 fold stratified cross validation
i=1
kf = KFold(n_splits=20, random_state=None, shuffle=True)
X=np.array(X)
y=np.array(y)
test=np.array(test)
y1=np.array(y1)

for train_index,test_index in kf.split(X):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X[train_index],X[test_index]
     ytr,yvl = y[train_index],y[test_index]
        
     model = LogisticRegression(random_state=1)
     model.fit(xtr, ytr)
     pred=model.predict_proba(xvl)[:,1]
     score = roc_auc_score(yvl,pred)
     print('roc_auc_score',score)
     i+=1


# In[43]:


pred=model.predict_proba(test)[:,1]
score = roc_auc_score(y1,pred)
print('roc_auc_score',score)

