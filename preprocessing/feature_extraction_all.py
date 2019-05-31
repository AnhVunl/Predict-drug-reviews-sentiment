# Load the packages

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

# Remove missing values and set up data frame
df_1 = pd.read_csv('/drugsComTrain_raw.tsv',delimiter='\t',encoding='utf-8')
df_2 = pd.read_csv('/drugsComTest_raw.tsv',delimiter='\t',encoding='utf-8')
df_1 = df_1.dropna()
df_2 = df_2.dropna()
df = pd.concat([df_1, df_2])

# Label sentiment: 0 (negative) if score is between 1 and 6, 1 (positive) if score is higher than 6
def scoring (x):
    if x >= 1 and x < 6:
        return 0
    elif x >= 6:
        return 1
df["sentiment"] = df["rating"].apply(scoring)
df = df[['review','sentiment', 'rating']] # we keep only the review, the sentiment and ratings for the analyses
df['review'] = df['review'].astype('str')
df.head()

# Feature extraction

# Positive verbs
def pos_verbs(review):
    only_letters = re.sub("[^a-zA-Z]", " ", review) # strip symbols and punctuations
    stop_words = set(stopwords.words('english')) 
    tokens = nltk.word_tokenize(only_letters.lower()) # tokenize the reviews
    tagged = nltk.pos_tag(tokens) # get the POS tags for each token
    lemmatizer = WordNetLemmatizer() # lemmatize text
    
    verbs = [] # list to keep all the positive verbs
    # filter out tokens that are not verbs and negative verbs
    for tup in tagged:
        
        if "VB" in tup[1]:
            synset = list(swn.senti_synsets(tup[0], 'v')) 
            # only keep the positive verbs (determined by SentiWordNet)
            if len(synset) > 1 and (synset[0].pos_score() > synset[0].neg_score() or synset[0].pos_score() >= synset[0].obj_score()):
                
                verbs.append(tup[0])
                            
    lower_case = [l.lower() for l in verbs] #lower case all the positive verbs
    
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))#lower_case)) # keep tokens that are not stopwords
    
    lemmas = []
    
    for t in filtered_result:
        word = lemmatizer.lemmatize(t, 'v')
        if word not in lemmas:
            lemmas.append(word)
     
    string = ""
    for word in lemmas: 
        string += word + " "
    return string
df['ps_verb'] = df['review'].apply(pos_verbs) # apply to all rows in our current data frame

# Negative verbs: same process as positive verbs, only difference is in the SWN score extraction
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
             # only keep the negative verbs (determined by SentiWordNet)
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
    string = ""
    for word in lemmas: 
        string += word + " "
    return string
df['neg_verb'] = df['review'].apply(neg_verbs)

# Combine both positive and negative verbs

df['verbs'] = df['ps_verb'].str.cat(df['neg_verb'], sep =" ") 

# Positive adjectives
def pos_adjectives(review):
    only_letters = re.sub("[^a-zA-Z]", " ",review) 
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
    
    string = ""
    for word in lemmas: 
        string += word + " "
    return string
df['ps_adj'] = df['review'].apply(pos_adjectives)

# Negative adjectives
def neg_adjectives(review):
    only_letters = re.sub("[^a-zA-Z]", " ", review) 
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
    
    string = ""
    for word in lemmas: 
        string += word + " "
    return string
df['neg_adj'] = df['review'].apply(neg_adjectives)

# Combine both positive and negative adjectives
df['adjectives'] = df['ps_adj'].str.cat(df['neg_adj'], sep =" ") 

# Positive adjective phrases: we'll use spaCy to make pairs of adjectives and nouns
def pos_np (text):
    doc = "" # final output of this function
    sentences = nlp(text) # this built-in function helps to tokenize and add POS tags to the review

    for token in sentences:
        if token.pos_ == 'NOUN': 
            for child in token.children: # a positive adjective phrase would have the root as a the noun and its child would be an adjective
                if child.pos_ == "ADJ": 
                    synset = list(swn.senti_synsets(child.text, 'a'))
                    if len(synset) > 1 and synset[0].pos_score() > synset[0].neg_score() and synset[0].pos_score() >= synset[0].obj_score():
                        doc += str(child.text.lower()) + " " + str(token.text.lower())
    return doc

df["pos_adj_phrase"] = df["review"].apply(pos_np)

# Negative adjectives phrases
def neg_np (text):
    def pos_np (text):
    doc = ""
    sentences = nlp(text)

    for token in sentences:
        if token.pos_ == 'NOUN':
            for child in token.children:
                if child.pos_ == "ADJ": 
                    synset = list(swn.senti_synsets(child.text, 'a'))
                    if len(synset) > 1 and synset[0].neg_score() > synset[0].pos_score() and synset[0].pos_score() >= synset[0].obj_score():
                        doc += str(child.text.lower()) + " " + str(token.text.lower())
    return doc
df["neg_adj_phrase"] = df["review"].apply(neg_np)

# Combine both positive and negative adjective phrases
df['adj_phrase'] = df['pos_adj_phrase'].str.cat(df['neg_adj_phrase'], sep =" ") 

# Linguistic rule: Negations (negation + adjective)
# Here we'll try to capture negation terms, meaning pairs of negative term-adjective, for example, "not good", "isn't effective"

negation = ["no", "haven't", "havent", "hasn", "weren't", "shan't", "shouldn", "mightn't", "hasn't", 
"couldn't", "aren't", "doesn't", "isn", "needn't", "hadn't", "didn't", "isn't", 
"doesn", "wasn't", "not", "aren", "mightn", "won't", "mustn", "mustn't", "wouldn", "wouldn't", "couldn", "nor", 
"needn", "shouldn't", "don't", "haven", "won", "t", "hadn", "didn", "dont", "none", "never", "nothing", "nor",
"nowhere", "not any", "neither", "dont", "n't"] # list of negative words

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

df["polarity_score"] = df["review"].apply(negative)

# Save it to a separate CSV file for the analysis

df.to_csv("processed_features.csv", sep='\t', na_rep = "", index = False)

# SUMMARY STATISTICS OF ALL FEATURE GROUPS

# ADJECTIVES
# total features of negative adjective: 894
# total features of positive adjectives: 813
# ratio of positive adjective per review: 1.2 words (~1 word)
# ratio of negative adjective per review: 1.5 words (~2 words)
# ratio of reviews without any emotion-carrying adjectives: 33386/213869 = 15.6%

# VERBS
# total features of negative verbs: 989
# total features of positive verbs: 932
# ratio of positive verbs per review: 1.2 words (~1 word)
# ratio of negative verbs per review: ~1 word
# ratio of reviews without any emotion-carrying verbs: 39153/213869 = 18.3% 

# NOUN PHRASES
# total features of positive noun phrases (2-word): 4705/2
# total features of positive noun phrases (2-word): 5602/2
# ratio of reviews without any pairs of emotion-carrying noun phrases: 112149/213869 = 52%

# NEGATION
# count of reviews without any negations: 43.6%
