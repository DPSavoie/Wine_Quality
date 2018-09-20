
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib


# In[7]:


data = pd.read_csv(r'D:\wine quality database\winequality-red.csv', sep=';')


# In[8]:


data.head()


# In[10]:


data.isnull().any()


# In[13]:


data.info()


# In[15]:


data.shape


# In[22]:


#prediction variable is quality specified 

y = data.quality
X = data.drop('quality', axis=1)


# In[23]:


#Split the test and training data, specific random state as random (123) and statify the sample by the target variable to training and test sets are similar
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                             test_size=0.2,
                                             random_state=123,
                                             stratify=y)


# In[27]:


#create a scler object to save the meand stdev of the training set

scaler = preprocessing.StandardScaler().fit(X_train)


# In[33]:


#print the saved means and stdev of the training set
X_train_scaled = scaler.transform(X_test)

print (X_train_scaled.mean(axis=0))

print (X_train_scaled.std(axis=0))


# In[34]:


#create a modelling pipeline that transforms the data using standard scaler and fits the model to random forest regressor

pipeline = make_pipeline(preprocessing.StandardScaler(),
                        RandomForestRegressor(n_estimators=100))


# In[36]:


#declare hyperparameters that decide whether the random forest creates a branch in the tree or not

print (pipeline.get_params())


# In[38]:


#these hyperarameters are chosen from above and pick the ones to tune through cross validation
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5,3, 1]}


# In[39]:


# These are the steps for CV:

# Split your data into k equal parts, or "folds" (typically k=10).
# Train your model on k-1 folds (e.g. the first 9 folds).
# Evaluate it on the remaining "hold-out" fold (e.g. the 10th fold).
# Perform steps (2) and (3) k times, each time holding out a different fold.
# Aggregate the performance across all k folds. This is your performance metric.


# In[40]:


#sklearn cross validation 
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

#Fit and tune model 
clf.fit(X_train, y_train)


# In[41]:


#best parameters to tune cv model on
print (clf.best_params_)


# In[43]:


print (clf.refit)


# In[44]:


#create a ne data set 
y_pred = clf.predict(X_test)


# In[45]:


#evaluate model permformance from metrics we imported earlier
print (r2_score(y_test, y_pred))

print (mean_squared_error(y_test, y_pred))


# In[59]:


import matplotlib.pyplot as plt
import seaborn as sns

correlation = data.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidth=0, vmin=1, cmap="Accent")


# In[68]:


#Decision tree classifier 
# specific prediction variable 

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                             test_size=0.2,)


# In[70]:


print(X_train.head())


# In[62]:


#preprocessing trained data 

X_train_scaled = preprocessing.scale(X_train)
print (X_train_scaled)


# In[66]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[67]:


confidence = clf.score(X_test, y_test)
print(confidence)


# In[71]:


y_pred = clf.predict(X_test)


# In[76]:


#converting the numpy array to list, take the first 5 entries and compare them 
x=np.array(y_pred).tolist()

#printing first 5 predictions
print("\nThe prediction:\n")
for i in range(0,5):
    print (x[i])
    
#printing first five expectations
print("\nThe expectation:\n")
print (y_test.head())


# In[77]:


#evaluate model permformance from metrics we imported earlier
print (r2_score(y_test, y_pred))

print (mean_squared_error(y_test, y_pred))


# In[81]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[82]:


#making a binary classification to increase accuracy, recoding needed

bins = (2, 6.5, 8)
group_names = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)


# In[84]:


label_quality = LabelEncoder()


# In[85]:


data['quality'] = label_quality.fit_transform(data['quality'])


# In[86]:


data['quality'].value_counts()


# In[88]:


#Now seperate the dataset as response variable and feature variabes
X = data.drop('quality', axis = 1)
y = data['quality']


# In[89]:


#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[90]:


#Applying Standard scaling to get optimized result
sc = StandardScaler()


# In[91]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[92]:


#updated random forest
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[93]:


print(classification_report(y_test, pred_rfc))


# In[94]:


print(confusion_matrix(y_test, pred_rfc))

