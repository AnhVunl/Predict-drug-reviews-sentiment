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

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

# Set up data frame

data = pd.read_csv('/processed_features.csv',delimiter='\t',encoding='utf-8')
data = data.replace(np.nan, '', regex=True)
data.head()

# Text feature transformation
# For each text variable, I'd change the X, for example, data['verb'], data['adjectives']

# Feature transformation 
vectorizer = TfidfVectorizer() # could also use CountVectorizer() as another word embedding method
X = vectorizer.fit_transform(data['verbs'])
y = data['sentiment']

# Feature selection:
feature_names = vectorizer.get_feature_names()
ch2 = SelectKBest(chi2, k = 100) # select the most 100 informative words
X_new = ch2.fit_transform(X, y)

# Access list of features
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

# Numerical feature transformation

X = pd.DataFrame(data['negation'])
y = data['sentiment']
X_2 = preprocessing.StandardScaler().fit(X).transform(X.astype('float'))

# Train, test, split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)
print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

# Training the model

knn = KNeighborsClassifier(n_neighbors = 19, weights = 'distance')
knn.fit(X_train, y_train)
y_pred_val = knn.predict(X_val)
y_pred_roc = knn.predict_proba(X_val)[:,1]

# Evaluate on validation set

print("Precision", precision_score(y_val, y_pred_val))
print("Recall", recall_score(y_val, y_pred_val))
print("ROC", roc_auc_score(y_val, y_pred_roc))
print("F1 score", f1_score(y_val, y_pred_val))

# GridSearch CV

parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
knn_grid = GridSearchCV(knn, parameters, scoring = 'roc_auc') # improve on the ROC_AUC metric since it has the lowest score
knn_grid.fit(X_train, y_train)

# Combine validation and train and refit the model with the new hyperparameter settings

X_final = vstack([X_val, X_train]).toarray()
y_train = pd.Series(y_train)
f = [y_val, y_train]
y_final = pd.concat(f)
print(X_final.shape)
print(y_final.shape)

# Re-fit our model

knn_grid.fit(X_final, y_final)

# Finally evaluate on the test set
y_pred_roc = knn_grid.predict_proba(X_test)[::, 1]
y_pred = knn_grid.predict(X_test)

print("Precision", precision_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("ROC", roc_auc_score(y_test, y_pred_roc))
print("F1 score", f1_score(y_test, y_pred))

# Cross-validation on training set

result = cross_val_score(knn_grid, X_train, y_train, cv = 5, scoring = 'recall')
print(result.mean())
