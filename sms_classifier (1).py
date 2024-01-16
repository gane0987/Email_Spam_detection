#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[41]:


raw_mail_data=pd.read_csv('mail_data.csv')


# In[42]:


print(raw_mail_data)


# In[43]:


#replace a null values into null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[44]:


#printng the first five rows of data frame
mail_data.head()


# In[45]:


#checking no of rows here
mail_data.shape


# In[46]:


#labeling spam and ham has o and 1
mail_data.loc[mail_data['Category']=='spam','Category',]=0
mail_data.loc[mail_data['Category']=='ham','Category',]=1


# In[47]:


#separating texts and labels
X=mail_data['Message']
Y=mail_data['Category']


# In[48]:


#spliting the datas into training and for testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
print(X.shape)
print(X_train.shape)
print(Y_test.shape)


# In[49]:


#transforming the text data to feature vectors using logistic regression
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

#convert Y train and test as integers
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# In[18]:


print(X_train_features)


# In[50]:


#training with the logistic regression
model=LogisticRegression()
model.fit(X_train_features,Y_train)


# In[51]:


#Prediction on training data
prediction1=model.predict(X_train_features)
accuracy_on_training=accuracy_score(Y_train,prediction1)


# In[52]:


print('Accuracy on train data :',accuracy_on_training)


# In[53]:


#Prediction on test data
prediction_on_testing=model.predict(X_test_features)
accuracy_on_testing=accuracy_score(Y_test,prediction_on_testing)


# In[54]:


print('Accurcy on test data:',accuracy_on_testing)


# In[55]:


input_mail=['URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18']
input_data_features=feature_extraction.transform(input_mail)
prediction2=model.predict(input_data_features)
print(prediction2)


# In[56]:


if prediction[0]==1:
    print('Ham mail')
else:
    print('Spam mail')


# In[ ]:




