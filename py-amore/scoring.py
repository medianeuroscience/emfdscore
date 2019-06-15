from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('english')
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import spacy
import re, fnmatch
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = set(list(nltk_stopwords) + list(ENGLISH_STOP_WORDS) + list(STOP_WORDS))
from collections import Counter
from load_mfds import * 
import progressbar, time


def tokenizer(doc):
    
    '''Performs minimal preprocessing on textual document.
    Steps include tokenization, lower-casing, and 
    stopword/punctuation/whitespace removal. 
    Returns list of processed tokens'''
    
    return  [x.lower_ for x in doc if x.lower_ not in stopwords and not x.is_punct and not x.is_digit and not x.is_quote and not x.like_num and not x.is_space] 


def score_emfd(doc):
    
    '''Scores documents with the e-MFD.'''
    
    emfd_score = {k:0 for k in probabilites+senti}
    moral_words = [ emfd[token] for token in doc if token in emfd.keys() ]
    
    for dic in moral_words:
        emfd_score['care_p'] += dic['care_p']
        emfd_score['fairness_p'] += dic['fairness_p']
        emfd_score['loyalty_p'] += dic['loyalty_p']
        emfd_score['authority_p'] += dic['authority_p']
        emfd_score['sanctity_p'] += dic['sanctity_p']
        
        emfd_score['care_sent'] += dic['care_sent']
        emfd_score['fairness_sent'] += dic['fairness_sent']
        emfd_score['loyalty_sent'] += dic['loyalty_sent']
        emfd_score['authority_sent'] += dic['authority_sent']
        emfd_score['sanctity_sent'] += dic['sanctity_sent']
    
    emfd_score = {k:v/len(doc) for k,v in emfd_score.items()}
    nonmoral_words = len(doc)-len(moral_words)
    emfd_score['moral_nonmoral_ratio'] =  len(moral_words)/nonmoral_words 
    
    return emfd_score


def score_mfd(doc):
    
    '''Scores documents with the original MFD.'''
    
    mfd_score = {k:0 for k in mfd_foundations}
    moral_words = []
    for token in doc:
        for v in mfd_regex.keys():
            if mfd_regex[v].match(token):
                for f in mfd[v]:
                    mfd_score[f] += 1
    
    mfd_score = {k:v/len(doc) for k,v in mfd_score.items()}
    
    return mfd_score


def score_mfd2(doc):
    
    '''Scores documents with the MFD2.'''
    
    mfd2_score = {k:0 for k in mfd2_foundations}
    moral_words = [ mfd2[token]['foundation'] for token in doc if token in mfd2.keys() ]
    f_counts = Counter(moral_words)
    mfd2_score.update(f_counts)    
    mfd2_score = {k:v/len(doc) for k,v in mfd2_score.items()}
    
    return mfd2_score


def score_docs(csv, dic_type, num_docs):
    
    '''Wrapper function that executes functions for preprocessing and dictionary scoring. 
    dict_type specifies the dicitonary with which the documents should be scored.
    Accepted values are: [emfd, mfd, mfd2]'''
    
    nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
    nlp.add_pipe(tokenizer, name="mfd_tokenizer")
    
    if dic_type == 'emfd':
        nlp.add_pipe(score_emfd, name="score_emfd", last=True)
    elif dic_type == 'mfd':
        nlp.add_pipe(score_mfd, name="score_mfd", last=True)
    elif dic_type == 'mfd2':
        nlp.add_pipe(score_mfd2, name="score_mfd2", last=True)
    else:
        print('Dictionary type not recognized. Available values are: emfd, mfd, mfd2')
        return 
    
    scored_docs = []
    with progressbar.ProgressBar(max_value=num_docs) as bar:
        for i, row in csv[0].iteritems():
            scored_docs.append(nlp(row))
            bar.update(i)

    df = pd.DataFrame(scored_docs)
    if dic_type == 'emfd':
        df['f_var'] = df[probabilites].var(axis=1)
        df['sent_var'] = df[senti].var(axis=1)
        
    return df