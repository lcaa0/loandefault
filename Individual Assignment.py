#!/usr/bin/env python
# coding: utf-8

# In[161]:


import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_tree


# In[162]:


df = pd.read_csv("Credit Card Default II (balance).csv")


# In[163]:


df


# In[164]:


df.info()


# In[165]:


print(df.describe())


# # Visualisation and Data Cleaning

# In[166]:


df.hist(bins=30, figsize=(15, 10))


# In[167]:


df.boxplot()


# In[168]:


df.sort_values("age").head(5)


# In[169]:


df2 = df[df['age'] > 0]  


# In[170]:


df2
      


# In[171]:


Y = df2.loc[:, ["default"]]


# In[172]:


X = df2.iloc[:,0:3]


# # Decision Tree Classifier - Cart

# In[173]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


# In[174]:


from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix


# In[175]:


clfmodel = tree.DecisionTreeClassifier(max_depth=7)
clfmodel.fit(X_train,Y_train)
pred = clfmodel.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print("Confusion Matrix:")
print(cm)
accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
print(accuracy)
print( "\n Classification Report")
print(classification_report(Y_test, pred))


# In[193]:


plt.figure(figsize=(30,20))
tree.plot_tree(clfmodel, fontsize=10)
plt.show()


# In[176]:


X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_train,Y_train)


# In[177]:


for i in range(1,100):
    model = tree.DecisionTreeClassifier(max_depth=i+2)
    model.fit(X_train1, Y_train1)
    pred = model.predict(X_test1)
    cm = confusion_matrix(Y_test1, pred)
    accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
    print(i)
    print(accuracy)
    print("========================")
    print(i)
    print(accuracy)


# In[178]:


for i in range(1,100):
    model = tree.DecisionTreeClassifier(min_samples_split=i+2)
    model.fit(X_train1, Y_train1)
    pred = model.predict(X_test1)
    cm = confusion_matrix(Y_test1, pred)
    accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
    print(i)
    print(accuracy)
    print("========================")


# # Logistic Regression

# In[179]:


from sklearn import linear_model

logregmodel = linear_model.LogisticRegression()
logregmodel.fit(X_train,Y_train)
pred = logregmodel.predict(X_train)
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
print(accuracy)

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
print(accuracy)
print( "\n Classification Report")
print(classification_report(Y_test, pred))


# # Random Forest Classifier

# In[180]:


from sklearn.ensemble import RandomForestClassifier
rfcmodel = RandomForestClassifier(max_depth=7)
rfcmodel.fit(X_train, Y_train)
pred = rfcmodel.predict(X_test)

cm = confusion_matrix(Y_test, pred)
print("Confusion Matrix:")
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)
print( "\n Classification Report")
print(classification_report(Y_test, pred))


# 
# # Gradient Boosting Classifier

# In[181]:


from sklearn.ensemble import GradientBoostingClassifier
gbcmodel = GradientBoostingClassifier(max_depth=7)
gbcmodel.fit(X_train, Y_train)
pred = gbcmodel.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print("Confusion Matrix:")
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)
print( "\n Classification Report")
print(classification_report(Y_test, pred))


# In[182]:


fr = gbcmodel.feature_importances_
print(fr)


# In[ ]:




