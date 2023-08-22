#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


df=pd.read_csv("full_data.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df['gender'].value_counts()


# In[7]:


df['age'].groupby(df['gender']).mean()


# In[8]:



sns.heatmap(df.corr(),cmap='coolwarm',annot=True)


# In[9]:


px.histogram(df,x='gender',y='age',color='ever_married')


# In[10]:


df['avg_glucose_level'].groupby(df['gender']).mean()


# In[11]:


px.scatter(df,x='avg_glucose_level',y='bmi',color='gender')


# In[12]:


df['stroke'].value_counts()


# In[13]:


sns.countplot(data=df,x='stroke')


# In[14]:


px.histogram(df,x='gender',y='stroke')


# In[15]:


sns.histplot(x='age', data=df, hue='stroke', kde=True)


# In[16]:


px.histogram(df,x='age',color='stroke')


# In[17]:


px.histogram(df,x='smoking_status',color='stroke')


# In[18]:


px.histogram(df,x='heart_disease',color='stroke')


# In[19]:


px.histogram(df,x='hypertension',color='stroke')


# In[20]:


from sklearn.preprocessing import LabelEncoder
l1=LabelEncoder()


# In[21]:


df["gender"]=l1.fit_transform(df["gender"])
df["ever_married"]=l1.fit_transform(df["ever_married"])
df["work_type"]=l1.fit_transform(df["work_type"])
df["Residence_type"]=l1.fit_transform(df["Residence_type"])
df["smoking_status"]=l1.fit_transform(df["smoking_status"])


# In[22]:


df.head()


# In[23]:


df.describe().transpose()


# In[47]:


# x=df.iloc[:, :-1].values
# y=df.iloc[:, -1].values
# x=df.drop('stroke',axis=1)
# y=df['stroke']
x=df.drop(["stroke"],axis=1).values
y=df["stroke"]


# In[48]:


x


# In[49]:


y


# In[75]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


# In[78]:


sc = StandardScaler()
X = sc.fit_transform(x)
x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.2)


# In[79]:


x_train.shape, x_test.shape


# In[80]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
print("accuray is " , lr.score(x_test , y_test))


# In[73]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
prediction=lr.predict(x_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


# In[55]:


print("Logistic Regression accuray is {:.2f}%" .format(lr.score(x_test , y_test)*100) )


# In[56]:


from sklearn.neighbors import KNeighborsClassifier


# In[34]:


error_rate=[]
for i in range (1,50):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred=knn.predict(x_test)
    error_rate.append(np.mean(pred != y_test))


# In[35]:


plt.figure(figsize=(14,6))
plt.plot(range(1,50),error_rate,color='blue',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate Vs K Value')
plt.ylabel('Error Rate')
plt.xlabel('K Value')


# In[36]:


knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[37]:


print("The accuracy of the KNN Model is {:.2f}%".format(knn.score(x_test,y_test)*100))


# In[38]:


from sklearn.tree import DecisionTreeClassifier


# In[39]:


dtree= DecisionTreeClassifier(criterion='entropy')
dtree.fit(x_train,y_train)
prediction=dtree.predict(x_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


# In[40]:


print("The accuracy of the Decicion Tree Model is {:.2f}%".format(dtree.score(x_test,y_test)*100))


# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:





# In[42]:


rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[43]:


print("The accuracy of the Random Forest Model is {:.2f}%".format(rfc.score(x_test,y_test)*100))


# In[44]:


from sklearn.svm import SVC


# In[45]:


svm=SVC()
svm.fit(x_train,y_train)
svm.fit(x_train,y_train)
prediction=svm.predict(x_test)
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


# In[46]:


print("The accuracy of the Support Vector Machine Model is {:.2f}%".format(svm.score(x_test,y_test)*100))


# In[ ]:




