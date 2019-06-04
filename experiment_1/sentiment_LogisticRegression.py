# Loading packages
import numpy
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
from scipy.sparse import csr_matrix, find, hstack, vstack

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

# Set up data frame

data = pd.read_csv('processed_features.csv',delimiter='\t',encoding='utf-8')
data = data.replace(np.nan, '', regex=True)

# Text feature transformation
# For each text variable, I'd change the X, for example, data['verb'], data['adjectives']

# Feature transformation 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['verbs'])
y = data['sentiment']

# Feature selection:
feature_names = vectorizer.get_feature_names()
ch2 = SelectKBest(chi2, k = 100)
X_new = ch2.fit_transform(X, y)

# List of selected features:
selected_feature_names = [feature_names[i] for i in ch2.get_support(indices=True)] 
chi2_scores = pd.DataFrame(list(zip(selected_feature_names, ch2.scores_)), 
columns = ['term', 'score'])
fr = pd.DataFrame(chi2_scores.sort_values(['score'], ascending = False))

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = []
for word in fr.term:
    x = lemmatizer.lemmatize(word, 'v')
    if x not in words:
        words.append(x)
print(words) # List of selected features

# Numerical feature transformation

X = pd.DataFrame(data['negation'])
y = data['sentiment']
X_2 = preprocessing.StandardScaler().fit(X).transform(X.astype('float'))

# Train, test, split

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Train the model

lgm = LogisticRegression()
lgm.fit(X_train, y_train)

y_pred_lr = lgm.predict(X_val)
y_pred_val = lgm.predict_proba(X_val)[:,1]

# Evaluate on the validation set

print("Precision", precision_score(y_val, y_pred_lr))
print("Recall", recall_score(y_val, y_pred_lr))
print("ROC", roc_auc_score(y_val, y_pred_val))
print("F1 score", f1_score(y_val, y_pred_lr))

# Tune hyperparameters

parameters = {'penalty':['l1','l2'], 'C':[0.001, 0.01, 0.1, 1,10]}
lgm_grid = GridSearchCV(lgm, parameters, scoring = 'roc_auc')
lgm_grid.fit(X_train, y_train)

# Combine validation and train and refit the model with the new hyperparameter settings

X_final = vstack([X_val, X_train]).toarray()
y_train = pd.Series(y_train)
f = [y_val, y_train]
y_final = pd.concat(f)
print(X_final.shape)
print(y_final.shape)

# Let's fit the optimized model

lgm_grid.fit(X_final, y_final)

y_pred_roc = lgm_grid.predict_proba(X_test)[::, 1]
auc = roc_auc_score(y_test, y_pred_roc)
y_pred = lgm_grid.predict(X_test)

# Evaluate on test set

print("Precision", precision_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("ROC", roc_auc_score(y_test, y_pred_roc))
print("F1 score", f1_score(y_test, y_pred))


# Cross-validation on training set

result = cross_val_score(lgm_grid, X_train, y_train, cv = 5, scoring = 'recall')
print(result.mean())
