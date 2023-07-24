#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction

# ## Binary Classification using Logistic Regression![download.jpg](attachment:download.jpg)

# ## Importing Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## Importing & Loading the dataset

# In[3]:


df = pd.read_csv('train.csv')
df.head()


# ## Dataset Info:-

# In[4]:


df.info()


# ## Database shape:-

# In[5]:


df.shape


# # Data Cleaning
# ## Checking the Missing Values

# In[6]:


df.isnull().sum()


# ### We will fill the Missing Values in "LoanAmount" & "Credit_History" by the 'Mean' & 'Median' of the respective variables.

# In[7]:


df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())


# In[8]:


df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())


# ### Let's confirm if there are any missing values in 'LoanAmount' & 'Credit_History'

# In[9]:


df.isna().sum()


# ## Let's drop all the missing values remaining.

# In[10]:


df.dropna(inplace=True)


# ## Check missing values final time!

# In[11]:


df.isnull().sum()


# Here, we have dropped all the missing values to avoid disturbance in the model. The Loan Prediction requires all the details to work efficiently thus the missing value are dropped.

# ## Let's check the final Dataset Shape

# In[12]:


df.shape


# ## Exploratory Data Analyis

# ### Comparison between Parameters in getting the Loan:

# In[45]:


plt.figure(figsize = (100, 50))
sns.set(font_scale = 5)
plt.subplot(331)
sns.countplot(x ='Gender',hue=df['Loan_Status'], data=df)

plt.subplot(332)
sns.countplot(x ='Married',hue=df['Loan_Status'], data=df)

plt.subplot(333)
sns.countplot(x ='Education',hue=df['Loan_Status'], data=df)

plt.subplot(334)
sns.countplot(x ='Self_Employed',hue=df['Loan_Status'], data=df)

plt.subplot(335)
sns.countplot(x ='Property_Area',hue=df['Loan_Status'], data=df)


# ## Let's replace the Variable values of Numeical form & display the Value Counts

# In[27]:


df['Loan_Status'].replace('Y', 1, inplace=True)
df['Loan_Status'].replace('N', 0, inplace=True)


# In[28]:


df['Loan_Status'].value_counts()


# In[17]:


df.Married=df.Married.map({'Yes':1,'No':0})
df['Married'].value_counts()


# In[16]:


df.Gender=df.Gender.map({'Male':1,'Female':0})
df['Gender'].value_counts()


# In[18]:


df.Dependents=df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
df['Dependents'].value_counts()


# In[19]:


df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
df['Education'].value_counts()


# In[20]:


df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
df['Self_Employed'].value_counts()


# In[21]:


df.Property_Area=df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
df['Property_Area'].value_counts()


# In[23]:


df['LoanAmount'].value_counts()


# In[24]:


df['Loan_Amount_Term'].value_counts()


# In[25]:


df['Credit_History'].value_counts()


# In[ ]:





# From the above figure, we can see that **Credit_History** (Independent Variable) has the maximum correlation with **Loan_Status** (Dependent Variable). Which denotes that the Loan_Status is heavily dependent on the Credit_History.

# In[34]:


df.head()


# In[35]:


df


# ### Importing Packages for Classification algorithms 

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ### splitting the data into Train and Test set

# In[38]:


X = df.iloc[1:542,1:12].values
y = df.iloc[1:542,12].values


# In[37]:


df.iloc[1:542,1:12].values


# In[40]:


df.iloc[1:542,12].values


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


# ### Logistic Regression (LR)

# Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable.
# 
# Mathematically, a logistic regression model predicts P(Y=1) as a function of X. It is one of the simplest ML algorithms that can be used for various classification problems such as spam detection, Diabetes prediction, cancer detection etc.
# 
# **Sigmoid Function**
# ![logistic%20regression.png](attachment:logistic%20regression.png)

# In[43]:


model = LogisticRegression()
model.fit(X_train,y_train)

lr_prediction = model.predict(X_test)
print('Logistic Regression accuracy = ', metrics.accuracy_score(lr_prediction, y_test))


# In[44]:


print('y_predicted', lr_prediction)
print('y_test',y_test)


# ### CONCLUSION:

# 1. The Loan Status is heavily dependent on the Credit History for Predictions.
# 2. The Logistic Regression algorithm gives us the maximum Accuracy (79% approx) compared to the other 3 Machine Learning   Classification Algorithms.

# **Complete Project on Github** : https://github.com/Vyas-Rishabh/Python_Loan_Prediction_using_Logistic_Regression
