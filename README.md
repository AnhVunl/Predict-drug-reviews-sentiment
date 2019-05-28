# sentiment_drug_reviews

# Loading packages
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
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import neighbors, datasets, preprocessing
from sklearn.preprocessing import StandardScaler
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



