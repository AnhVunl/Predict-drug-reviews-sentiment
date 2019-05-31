# Load packages

import numpy
import numpy as np
import pandas as pd
import scipy.stats as stats
import pylab as pl
import scipy
from scipy.sparse import csr_matrix, find, hstack, vstack
import pylab as pl
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


# Set up data frame

data = pd.read_csv('/full_set_1st_analysis.csv',delimiter='\t',encoding='utf-8')
data = data.replace(np.nan, '', regex=True)
data = data.drop(['polarity_score','count_pos_np','count_neg_np',
 'freq_neg_adj','freq_neg_verb','freq_ps_adj','freq_ps_verb','grams','lemma'], axis = 1)
data.head()


# Text feature transformation

# For each text variable, I'd change the X, for example, data['verb'], data['adjectives'] 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['verbs'])
y = data['sentiment']

# Feature selection

feature_names = vectorizer.get_feature_names()
ch2 = SelectKBest(chi2, k = 100)
X_new = ch2.fit_transform(X, y)

# List of selected features

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
print(verbs)


# Numerical feature transformation

X = pd.DataFrame(data['negation'])
y = data['sentiment']
X_2 = preprocessing.StandardScaler().fit(X).transform(X.astype('float'))

# Train, test, split

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Train the model

svm = LinearSVC().fit(X_train, y_train)
y_pred = svm.predict(X_val)

# Initial test on validation set
print("Precision", precision_score(y_val, y_pred))
print("Recall", recall_score(y_val, y_pred))
print("ROC", roc_auc_score(y_val, y_pred))
print("F1 score", f1_score(y_val, y_pred))


# Tune our hyperparameters

parameters = {'C':[0.001, 0.01, 0.1, 1, 10]}
svm_grid = GridSearchCV(svm, parameters, cv = 5, scoring = 'roc_auc')
svm_grid.fit(X_train, y_train)

svm_grid.best_params_
svm_grid.best_score_


# Combine validation and train and refit the model with the new hyperparameter settings

X_final = vstack([X_val, X_train]).toarray()
y_train = pd.Series(y_train)
f = [y_val, y_train]
y_final = pd.concat(f)
print(X_final.shape)
print(y_final.shape)

# Train the optimized model
svm_grid.fit(X_final, y_final)
y_pred_svm = svm_grid.predict(X_test)


# Evaluate

print("Precision", precision_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("ROC", roc_auc_score(y_test, y_pred))
print("F1 score", f1_score(y_test, y_pred))


# Cross-validation on training set
result = cross_val_score(lgm_grid, X_train, y_train, cv = 5, scoring = 'recall')
print(result.mean())
