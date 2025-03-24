#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv("weight_height_dataset.csv")
df


# In[7]:


print(df['Class'].value_counts())
sns.countplot(x='Class',data=df)
plt.show()


# In[8]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[10]:


print("x_train:",x_train.shape)
print("x_test:",x_test.shape)
print("y_train:",y_train.shape)
print("y_test:",y_test.shape)


# In[11]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_trainn=sc.fit_transform(x_train)
x_testt=sc.transform(x_test)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5, metric= 'minkowski',p=2)
classifier.fit(x_trainn,y_train)


# In[26]:


print(classifier.predict(sc.transform([[150,46]])))


# In[15]:


y_pred=classifier.predict(x_testt)
y_pred


# In[16]:


from sklearn.metrics import confusion_matrix
cf=confusion_matrix(y_test,y_pred)
cf


# In[19]:


labels=classifier.classes_
labels


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

fig,ax=plt.subplots(figsize=(8,6))

sns.heatmap(cf, annot=True, annot_kws={"size":20},fmt='d', cmap="Blues",xticklabels=labels,yticklabels=labels, ax=ax)

ax.set_title('confusion matrix')
ax.set_xlabel('predicted')
ax.set_ylabel('actual')

plt.show(block=False)


# In[27]:


from sklearn.metrics import classification_report,roc_curve,auc
print(classification_report(y_test,y_pred,target_names=labels))


# In[29]:


df = pd.DataFrame(x_train, columns=['Height(cm)','Weight(kg)'])
custom_palette = ['#FF5733', '#33FF57', '#3366FF']

df['Class'] = y_train
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Height(cm)', y='Weight(kg)', hue='Class', palette=custom_palette)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Scatter Plot of Training Set')
plt.legend(title='Class')
plt.grid(True)
plt.show()


# In[31]:


df = pd.DataFrame(x_test, columns=['Height(cm)','Weight(kg)'])
custom_palette = ['#FF5733', '#33FF57', '#3366FF']

df['Class'] = y_test
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Height(cm)', y='Weight(kg)', hue='Class', palette=custom_palette)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Scatter Plot of Test Set')
plt.legend(title='Class')
plt.grid(True)
plt.show()


# In[33]:


print("Train data accuracy:",accuracy_score(y_true =y_train, y_pred=classifier.predict(x_trainn)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=classifier.predict(x_testt)))


# In[35]:


neighbors=np.arange(1,11)
train_accuracies = []
test_accuracies = []
for k in neighbors:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_trainn,y_train)
    train_accuracy=knn.score(x_trainn,y_train)
    test_accuracy=knn.score(x_testt,y_test)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

plt.figure(figsize=(10, 6))
plt.plot(neighbors, train_accuracies, label='Train Accuracy')
plt.plot(neighbors, test_accuracies, label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:




