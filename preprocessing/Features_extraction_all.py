#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[1]:


import scipy
from scipy.sparse import csr_matrix, find, hstack
import numpy
import numpy as np
import pandas as pd
import scipy.stats as stats
import pylab as pl
import nltk, re
import spacy
nlp = spacy.load('en_core_web_sm')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('sentiwordnet')
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk import ngrams
from nltk.stem import WordNetLemmatizer

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import neighbors, datasets, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# # Loading data

# In[11]:


# Combining both train and test data for pre-processing
df_1 = pd.read_csv('/Users/anhvu/Thesis/drugsComTrain_raw.tsv',delimiter='\t',encoding='utf-8')
df_2 = pd.read_csv('/Users/anhvu/Thesis/drugsComTest_raw.tsv',delimiter='\t',encoding='utf-8')
df_1 = df_1.dropna()
df_2 = df_2.dropna()
df = pd.concat([df_1, df_2])


# In[12]:


# Label sentiment: 0 (negative) if score is between 1 and 5, 1 (positive) if score is higher than 5
def scoring (x):
    if x >= 1 and x < 6:
        return 0
    elif x >= 6:
        return 1
df["sentiment"] = df["rating"].apply(scoring)


# In[13]:


df = df[['review','sentiment', 'rating']] # we keep only the review, the sentiment and ratings for the analyses
df['review'] = df['review'].astype('str')


# In[28]:


df.head()


# ## Single feature set: positive verbs

# In[26]:


# Remove stopwords, punctuation and pre-processing our reviews
def pos_verbs(review):
    # text cleaning steps
    only_letters = re.sub("[^a-zA-Z]", " ", review) # strip of symbols and punctuations
    stop_words = set(stopwords.words('english')) 
    tokens = nltk.word_tokenize(only_letters.lower()) # tokenize the reviews
    tagged = nltk.pos_tag(tokens) # get the POS tags for each token
    lemmatizer = WordNetLemmatizer() # lemmatize text
    
    verbs = []
    # filter out tokens that are not verbs and negative verbs
    for tup in tagged:
        
        if "VB" in tup[1]:
            synset = list(swn.senti_synsets(tup[0], 'v')) 
        
            if len(synset) > 1 and (synset[0].pos_score() > synset[0].neg_score() or synset[0].pos_score() >= synset[0].obj_score()):
                
                # filter out verbs that have a positive score lower than objective score
                verbs.append(tup[0])
                            
    lower_case = [l.lower() for l in verbs] #lower case all the words
    
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))#lower_case)) # keep tokens that are not stopwords
    
    lemmas = []
    
    for t in filtered_result:
        word = lemmatizer.lemmatize(t, 'v')
        if word not in lemmas:
            lemmas.append(word)
    
    # create new output as strings
    string = ""
    for word in lemmas: 
        string += word + " "
    return string


# In[27]:


df['ps_verb'] = df['review'].apply(pos_verbs)


# # Single feature set: negative verbs

# In[29]:


# same process for negative verbs
def neg_verbs(review):
    only_letters = re.sub("[^a-zA-Z]", " ",review)
    stop_words = set(stopwords.words('english')) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    tagged = nltk.pos_tag(tokens)
    verbs = []
    wordnet_lemmatizer = WordNetLemmatizer() # lemmatize text

    for tup in tagged:
        if "VB" in tup[1]:
            synset = list(swn.senti_synsets(tup[0], 'v'))
            
            if len(synset) > 1 and (synset[0].neg_score() >= synset[0].obj_score() or synset[0].neg_score() > synset[0].pos_score()):
                if tup[0] not in verbs:
                    verbs.append(tup[0])
                
    lower_case = [l.lower() for l in verbs]

    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = []
    
    for t in filtered_result:
        word = lemmatizer.lemmatize(t, 'v')
        if word not in lemmas:
            lemmas.append(word)
    
    # create new output as strings
    string = ""
    for word in lemmas: 
        string += word + " "
    return string


# In[ ]:


df['neg_verb'] = df['review'].apply(neg_verbs)


# In[ ]:


df['verbs'] = df['ps_verb'].str.cat(df['neg_verb'], sep =" ") 


# # Single feature set: positive adjectives

# In[30]:


def pos_adjectives(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    tagged = nltk.pos_tag(tokens)
    
    adjectives = []
    stop_words = set(stopwords.words('english')) 

    wordnet_lemmatizer = WordNetLemmatizer() # lemmatize text

    for tup in tagged:
        if tup[1] == "JJ": # get the polarity score of all adjectives
            synset = list(swn.senti_synsets(tup[0], 'a'))
            # only include the positive adjectives - those that have a positive score lower than negative score or obbjective score would be excluded
            if len(synset) > 1 and (synset[0].pos_score() >= synset[0].obj_score() or synset[0].pos_score() > synset[0].neg_score()):
                if tup[0] not in adjectives:
                    adjectives.append(tup[0])
    
       
    lower_case = [l.lower() for l in adjectives]

    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = []
    
    for t in filtered_result:
        word = lemmatizer.lemmatize(t, 'a')
        if word not in lemmas:
            lemmas.append(word)
    
    # create new output as strings
    string = ""
    for word in lemmas: 
        string += word + " "
    return string


# In[ ]:


df['ps_adj'] = df['review'].apply(pos_adjectives)


# # Single feature set: negative adjectives

# In[31]:


def neg_adjectives(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    tagged = nltk.pos_tag(tokens)
    adjectives = []
    stop_words = set(stopwords.words('english')) 

    wordnet_lemmatizer = WordNetLemmatizer() # lemmatize text
    for tup in tagged:
        if tup[1] == "JJ":
            synset = list(swn.senti_synsets(tup[0], 'a'))
            if len(synset) > 1 and (synset[0].neg_score() >= synset[0].obj_score() or synset[0].pos_score() < synset[0].neg_score()):
                if tup[0] not in adjectives:
                    adjectives.append(tup[0])
        
    lower_case = [l.lower() for l in adjectives]

    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = []
    
    for t in filtered_result:
        word = lemmatizer.lemmatize(t, 'a')
        if word not in lemmas:
            lemmas.append(word)
    
    # create new output as strings
    string = ""
    for word in lemmas: 
        string += word + " "
    return string


# In[ ]:


df['neg_adj'] = df['review'].apply(neg_adjectives)


# # Multiple-word feature set: positive noun phrases

# In[32]:


## Make combinations of positive adjectives with nouns (list format)

def pos_np (text):
    stop_words = set(stopwords.words('english'))
    chunk = []
    doc = ""
    for word in text.split():
        if word not in stop_words:
            new +=" " + word    
    sentences = nlp(doc)
    for token in sentences:
        if token.pos_ == 'NOUN':
            for child in token.children:
                if child.pos_ == "ADJ": 
                    synset = list(swn.senti_synsets(child.text, 'a'))
                    if len(synset) > 1 and synset[0].pos_score() > synset[0].neg_score() and synset[0].pos_score() >= synset[0].obj_score():
                        chunk.append([str(child.text.lower()) + " " + str(token.text.lower())])
    return chunk


# In[ ]:


df["pos_noun_np"] = df["review"].apply(pos_np)


# # Multiple-word feature set: negative noun phrases

# In[33]:


def neg_np (text):
    stop_words = set(stopwords.words('english'))
    
    chunk = []
    
    doc = ""
         
    for word in text.split():
        if word not in stop_words:
            new +=" " + word    
        
    sentences = nlp(doc)
    
    for token in sentences:
        
        if token.pos_ == 'NOUN':
            for child in token.children:
                if child.pos_ == "ADJ": 
                    synset = list(swn.senti_synsets(child.text, 'a'))
                    
                    if len(synset) > 1 and synset[0].neg_score() > synset[0].pos_score() and synset[0].neg_score() >= synset[0].obj_score():
                        chunk.append([str(child.text.lower()) + " " + str(token.text.lower())])
   
    return chunk


# In[ ]:


df["neg_noun_np"] = df["review"].apply(neg_np)


# # Multiple-word feature set: bigrams+trigrams

# In[34]:


def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    stop_words = set(stopwords.words('english')) 

    wordnet_lemmatizer = WordNetLemmatizer() # lemmatize text
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

from nltk import ngrams
def ngrams(input_list):
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams


# In[ ]:


df['grams'] = df.lemma.apply(ngrams)


# # Linguistic rule: Negations (negation + adjective)

# In[35]:


negation = ["no", "haven't", "havent", "hasn", "weren't", "shan't", "shouldn", "mightn't", "hasn't", 
"couldn't", "aren't", "doesn't", "isn", "needn't", "hadn't", "didn't", "isn't", 
"doesn", "wasn't", "not", "aren", "mightn", "won't", "mustn", "mustn't", "wouldn", "wouldn't", "couldn", "nor", 
"needn", "shouldn't", "don't", "haven", "won", "t", "hadn", "didn", "dont", "none", "never", "nothing", "nor",
"nowhere", "not any", "neither", "dont", "n't"]

def negative (document):
    doc = nlp(document.strip(",;:?\!"))
    phrases = []
    score = 0 
    mean = 0
    total = 0
    for token in doc:
        if token.dep_ == "neg" or token.dep_ == "amod" or token.dep_ == "acomp":
            phrases.append(token.text)
    for word in phrases:
        synset = list(swn.senti_synsets(word, 'a'))  
        
        if len(synset) >= 1 and (synset[0].pos_score() > synset[0].neg_score() or synset[0].pos_score() >= synset[0].obj_score()):
            mean = (synset[0].pos_score())/(synset[0].pos_score() + synset[0].neg_score() + synset[0].obj_score())
            score = -1
            total += (mean* score)
            
        elif len(synset) >= 1 and (synset[0].neg_score() > synset[0].pos_score() or synset[0].neg_score() >= synset[0].obj_score()):
            mean = (synset[0].neg_score())/(synset[0].pos_score() + synset[0].neg_score() + synset[0].obj_score())
            score = 1
            total += (mean*score)
    return float(total)


# In[ ]:


df["polarity_score"] = df["review"].apply(negative)


# In[21]:


df.describe()


# # Overview subset after extracting all features
# def counting (review):
#     count = 0
#     for word in negation:
#         if word in review:
#             count +=1
#     return count
# data['count'] = data['review'].apply(counting)

# In[36]:


df.to_csv("processed_features.csv", sep='\t', na_rep = "", index = False)


# *ADJECTIVES*
# 
# total features of negative adjective: 894
# 
# total features of positive adjectives: 813
# 
# ratio of positive adjective per review: 1.2 words (~1 word)
# 
# ratio of negative adjective per review: 1.5 words (~2 words)
# 
# ratio of reviews without any emotion-carrying adjectives: 33386/213869 = 15.6%
# 
# *VERBS*
# 
# total features of negative verbs: 989
# 
# total features of positive verbs: 932
# 
# ratio of positive verbs per review: 1.2 words (~1 word)
# 
# ratio of negative verbs per review: ~1 word
# 
# ratio of reviews without any emotion-carrying verbs: 39153/213869 = 18.3% 
# 
# *NOUN PHRASES*
# 
# total features of positive noun phrases (2-word): 4705/2
# 
# total features of positive noun phrases (2-word): 5602/2
# 
# ratio of reviews without any pairs of emotion-carrying noun phrases: 112149/213869 = 52%
# 
# *NEGATION*
# 
# count of reviews without any negations: 43.6%
# 
# this function is not robust towards 'short phrasings', for example, 'not cool' or 'didn't work'. 

# # Combined positive & negative adjectives

# In[7]:


data.head()


# In[5]:


data = data.drop(['polarity_score'], axis = 1)


# In[45]:





# In[49]:


data.to_csv("full_set.csv", sep='\t', na_rep = "", index = False)


# In[2]:


data = pd.read_csv('/Users/anhvu/Thesis/full_set.csv',delimiter='\t',encoding='utf-8')
data = data.drop(['grams'], axis =1)
data = data.drop(['lemma'], axis =1)


# In[6]:


print(len(data))


# In[4]:


data = data.rename(columns = {'count_neg_noun_adj': 'neg_adj_np', 'count_neg_np': 'freq_neg_adj_np',
    'count_pos_noun_adj':'pos_adj_np','count_pos_np':'freq_pos_adj_np'})


# In[ ]:




