#!/usr/bin/env python
# coding: utf-8

# # Loading packages

# In[8]:


import pandas as pd
import scipy.stats as stats
import pylab as pl
import nltk, re
import spacy
nlp = spacy.load('en_core_web_sm')

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import scipy
from scipy.sparse import csr_matrix, find, hstack, vstack

import numpy
import numpy as np
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import pylab as pl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.listdir(os.getcwd())

from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import neighbors, datasets, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


# In[2]:


data = pd.read_csv('/Users/anhvu/Thesis/full_set_1st_analysis.csv',delimiter='\t',encoding='utf-8')
data = data.replace(np.nan, '', regex=True)
data = data.drop(['polarity_score','count_pos_np','count_neg_np',
 'freq_neg_adj','freq_neg_verb','freq_ps_adj','freq_ps_verb','grams','lemma'], axis = 1)


# In[3]:


data.head()


# # Text feature transformation

# In[ ]:


# For each text variable, I'd change the X, for example, data['verb'], data['adjectives']


# In[4]:


# Feature transformation 
vectorizer = CountVectorizer()
#vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['verbs'])
y = data['sentiment']


# In[5]:


# Feature selection:
feature_names = vectorizer.get_feature_names()
ch2 = SelectKBest(chi2, k = 100)
X_new = ch2.fit_transform(X, y)
selected_feature_names = [feature_names[i] for i in ch2.get_support(indices=True)] 
chi2_scores = pd.DataFrame(list(zip(selected_feature_names, ch2.scores_)), 
columns = ['term', 'score'])
fr = pd.DataFrame(chi2_scores.sort_values(['score'], ascending = False))

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
verbs = []
for word in fr.term:
    x = lemmatizer.lemmatize(word, 'v')
    if x not in verbs:
        verbs.append(x)
print(verbs) # List of selected features


# # Numerical feature transformation

# In[ ]:


X = pd.DataFrame(data['negation'])
y = data['sentiment']
X_2 = preprocessing.StandardScaler().fit(X).transform(X.astype('float'))


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2)
#X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size = 0.2)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[7]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)
print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)


# # Training the model

# In[14]:


knn = KNeighborsClassifier(n_neighbors = 19, weights = 'distance')
knn.fit(X_train, y_train)


# In[17]:


y_pred_val = knn.predict(X_val)
y_pred_roc = knn.predict_proba(X_val)[:,1]


# In[18]:


print("Precision", precision_score(y_val, y_pred_val))
print("Recall", recall_score(y_val, y_pred_val))
print("ROC", roc_auc_score(y_val, y_pred_roc))
print("F1 score", f1_score(y_val, y_pred_val))


# # GridSearch CV

# In[ ]:


parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
knn_grid = GridSearchCV(knn, parameters, scoring = 'roc_auc')
knn_grid.fit(X_train, y_train)


# # Combine validation and train and refit the model with the new hyperparameter settings

# In[25]:


X_final = vstack([X_val, X_train]).toarray()
y_train = pd.Series(y_train)
f = [y_val, y_train]
y_final = pd.concat(f)
print(X_final.shape)
print(y_final.shape)


# In[26]:


knn_grid.fit(X_final, y_final)


# In[27]:


y_pred_roc = knn_grid.predict_proba(X_test)[::, 1]
y_pred = knn_grid.predict(X_test)


# In[29]:


print("Precision", precision_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("ROC", roc_auc_score(y_test, y_pred_roc))
print("F1 score", f1_score(y_test, y_pred))


# # Cross-validation on training set

# In[30]:


result = cross_val_score(knn_grid, X_train, y_train, cv = 5, scoring = 'recall')
print(result.mean())


# In[ ]:




