import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction import text
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from collections import Counter
from emfdscore.load_mfds import *
import progressbar

try:
    nltk_stopwords = stopwords.words('english')
except:
    print('NLTK stopwords missing, downloading now.')
    nltk.download('stopwords')
    nltk_stopwords = stopwords.words('english')

stopwords = set(list(nltk_stopwords) + list(text.ENGLISH_STOP_WORDS) + list(STOP_WORDS))

# BoW Scoring #
@Language.component("mfd_tokenizer")
def tokenizer(doc):
    
    """Performs minimal preprocessing on textual document.
    Steps include tokenization, lower-casing, and 
    stopword/punctuation/whitespace removal. 
    Returns list of processed tokens"""
    
    return [x.lower_ for x in doc if x.lower_ not in stopwords and not x.is_punct and not x.is_digit and not x.is_quote and not x.like_num and not x.is_space]

@Language.component("score_emfd_all_sent")
def score_emfd_all_sent(doc):
    
    """Scores documents with the eMFD, where each word is assigned five probabilities and the average sentiment for each foundation 
    is returned."""

    # Initiate dictionary to store scores
    emfd_score = {k: 0 for k in probabilites+senti}

    # Collect e-MFD data for all moral words in document
    moral_words = [emfd[token] for token in doc if token in emfd.keys()]
    
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

    if len(moral_words) != 0:
        emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
        nonmoral_words = len(doc)-len(moral_words)
        try:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1
    else:
        emfd_score = {k: 0 for k in probabilites + senti}
        nonmoral_words = len(doc) - len(moral_words)
        try:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1
    
    return emfd_score

@Language.component("score_emfd_single_sent")
def score_emfd_single_sent(doc):
    
    """Scores documents with the eMFD, where each word is assigned one probability and the associated average sentiment score."""

    # Initiate dictionary to store scores
    emfd_score = {k: 0 for k in probabilites+senti}

    # Collect e-MFD data for all moral words in document
    moral_words = [emfd_single_sent[token] for token in doc if token in emfd_single_sent.keys()]
    
    for dic in moral_words:
        base_f = dic['foundation'].split('_')[0]
        emfd_score[dic['foundation']] += dic['score']
        emfd_score[base_f+'_sent'] += dic['sentiment']
        
      
    if len(moral_words) != 0:
        emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
        nonmoral_words = len(doc)-len(moral_words)
        try:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1
    else:
        emfd_score = {k: 0 for k in probabilites + senti}
        nonmoral_words = len(doc) - len(moral_words)
        try:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1
    
    return emfd_score

@Language.component("score_emfd_all_vice_virtue")
def score_emfd_all_vice_virtue(doc):
    
    """Scores documents with the eMFD, where each word is assigned ten vice-virtue scores."""

    # Initiate dictionary to store scores
    emfd_score = {k: 0 for k in mfd_foundations}

    # Collect e-MFD data for all moral words in document
    moral_words = [emfd_all_vice_virtue[token] for token in doc if token in emfd_all_vice_virtue.keys()]
    
    for dic in moral_words:
        for f in mfd_foundations:
            if f != 'moral':
                emfd_score[f] += dic[f]
        
    if len(moral_words) != 0:
        emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
        nonmoral_words = len(doc)-len(moral_words)
        try:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1
    else:
        emfd_score = {k: 0 for k in mfd_foundations}
        nonmoral_words = len(doc) - len(moral_words)
        try:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1
    
    return emfd_score

@Language.component("score_emfd_single_vice_virtue")
def score_emfd_single_vice_virtue(doc):
    
    """Scores documents with the eMFD, where each word is assigned one vice-virtue score."""

    # Initiate dictionary to store scores
    emfd_score = {k:0 for k in mfd_foundations if k !='moral'}

    # Collect e-MFD data for all moral words in document
    moral_words = [emfd_single_vice_virtue[token] for token in doc if token in emfd_single_vice_virtue.keys()]
    
    for dic in moral_words:
        emfd_score[dic['foundation']] += dic['score']

    if len(moral_words) != 0:
        emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
        nonmoral_words = len(doc)-len(moral_words)
        try:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1
    else:
        emfd_score = {k: 0 for k in mfd_foundations}
        nonmoral_words = len(doc) - len(moral_words)
        try:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            emfd_score['moral_nonmoral_ratio'] = len(moral_words) / 1
    
    return emfd_score

@Language.component("score_mfd")
def score_mfd(doc):
    
    """Scores documents with the original MFD."""
    
    mfd_score = {k:0 for k in mfd_foundations}
    moral_words = 0
    
    for token in doc:
        for v in mfd_regex.keys():
            if mfd_regex[v].match(token):
                moral_words += 1
                for f in mfd[v]:
                    mfd_score[f] += 1

    if moral_words != 0:
        mfd_score = {k: v/moral_words for k, v in mfd_score.items()}
        nonmoral_words = len(doc)-moral_words
        try:
            mfd_score['moral_nonmoral_ratio'] = moral_words / nonmoral_words
        except ZeroDivisionError:
            mfd_score['moral_nonmoral_ratio'] = moral_words / 1
    else:
        mfd_score = {k:0 for k in mfd_foundations}
        nonmoral_words = len(doc) - moral_words
        try:
            mfd_score['moral_nonmoral_ratio'] = moral_words / nonmoral_words
        except ZeroDivisionError:
            mfd_score['moral_nonmoral_ratio'] = moral_words / 1
    
    return mfd_score

@Language.component("score_mfd2")
def score_mfd2(doc):
    
    """Scores documents with the MFD2."""
    
    mfd2_score = {k:0 for k in mfd2_foundations}
    moral_words = [mfd2[token]['foundation'] for token in doc if token in mfd2.keys()]
    f_counts = Counter(moral_words)
    mfd2_score.update(f_counts)    

    if len(moral_words) != 0:
        mfd2_score = {k: v/len(moral_words) for k,v in mfd2_score.items()}
        nonmoral_words = len(doc)-len(moral_words)
        try:
            mfd2_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            mfd2_score['moral_nonmoral_ratio'] = len(moral_words) / 1
    else:
        mfd2_score = {k: 0 for k in mfd2_foundations}
        nonmoral_words = len(doc) - len(moral_words)
        try:
            mfd2_score['moral_nonmoral_ratio'] = len(moral_words) / nonmoral_words
        except ZeroDivisionError:
            mfd2_score['moral_nonmoral_ratio'] = len(moral_words) / 1

    return mfd2_score


def score_docs(csv, dic_type, prob_map, score_type, out_metrics, num_docs):
    
    """Wrapper function that executes functions for preprocessing and dictionary scoring.
    dict_type specifies the dicitonary with which the documents should be scored.
    Accepted values are: [emfd, mfd, mfd2]"""

    if score_type == 'wordlist':
        widgets = [
            'Processed: ', progressbar.Counter(),
            ' ', progressbar.Percentage(),
            ' ', progressbar.Bar(marker='❤'),
            ' ', progressbar.Timer(),
            ' ', progressbar.ETA(),
        ]

        with progressbar.ProgressBar(max_value=num_docs, widgets=widgets) as bar:
            moral_words = []
            for i, row in csv[0].iteritems():
                if row in emfd.keys():
                    moral_words.append(emfd[row])
                else:
                    bar.update(i)
                    continue
        

            emfd_score = {k: 0 for k in probabilites+senti}

            # Collect e-MFD data for all moral words in document
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
                bar.update(i)

            emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
            emfd_score['cnt'] = len(moral_words)
            df = pd.DataFrame(pd.Series(emfd_score)).T
            df = df[['cnt']+probabilites+senti]
            return df

    if score_type == 'gdelt.ngrams':
        widgets = [
            'Processed: ', progressbar.Counter(),
            ' ', progressbar.Percentage(),
            ' ', progressbar.Bar(marker='❤'),
            ' ', progressbar.Timer(),
            ' ', progressbar.ETA(),
        ]

        with progressbar.ProgressBar(max_value=num_docs, widgets=widgets) as bar:
            moral_words = []
            word_frequncies = []
            for i, row in csv.iterrows():
                if row['word'] in emfd.keys():
                    moral_words.append( {'scores':emfd[row['word']], 'freq': row['freq']} )
                    word_frequncies.append(int(row['freq']))
                else:
                    bar.update(i)
                    continue
        

            emfd_score = {k: 0 for k in probabilites+senti}

            # Collect e-MFD data for all moral words in document
            for dic in moral_words:
                emfd_score['care_p'] += (dic['scores']['care_p'] * dic['freq'])
                emfd_score['fairness_p'] += (dic['scores']['fairness_p'] * dic['freq'])
                emfd_score['loyalty_p'] += (dic['scores']['loyalty_p'] * dic['freq'])
                emfd_score['authority_p'] += (dic['scores']['authority_p'] * dic['freq'])
                emfd_score['sanctity_p'] += (dic['scores']['sanctity_p'] * dic['freq'])
        
                emfd_score['care_sent'] += (dic['scores']['care_sent'] * dic['freq'])
                emfd_score['fairness_sent'] += (dic['scores']['fairness_sent'] * dic['freq'])
                emfd_score['loyalty_sent'] += (dic['scores']['loyalty_sent'] * dic['freq'])
                emfd_score['authority_sent'] += (dic['scores']['authority_sent'] * dic['freq'])
                emfd_score['sanctity_sent'] += (dic['scores']['sanctity_sent'] * dic['freq'])
                bar.update(i)

            emfd_score = {k: v/sum(word_frequncies) for k, v in emfd_score.items()}
            emfd_score['cnt'] = sum(word_frequncies)
            df = pd.DataFrame(pd.Series(emfd_score)).T
            df = df[['cnt']+probabilites+senti]
            return df

    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    nlp.add_pipe("mfd_tokenizer")
    
    if dic_type == 'emfd':
        if prob_map == 'all' and out_metrics == 'sentiment':
            nlp.add_pipe("score_emfd_all_sent", last=True)
        elif prob_map == 'all' and out_metrics == 'vice-virtue':
            nlp.add_pipe("score_emfd_all_vice_virtue", last=True)
        elif prob_map == 'single' and out_metrics == 'sentiment':
            nlp.add_pipe("score_emfd_single_sent", last=True)
        elif prob_map == 'single' and out_metrics == 'vice-virtue':
            nlp.add_pipe("score_emfd_single_vice_virtue", last=True)
    elif dic_type == 'mfd':
        nlp.add_pipe("score_mfd", last=True)
    elif dic_type == 'mfd2':
        nlp.add_pipe("score_mfd2", last=True)
    else:
        print('Dictionary type not recognized. Available values are: emfd, mfd, mfd2')
        return 

    scored_docs = []
    widgets = [
        'Processed: ', progressbar.Counter(),
        ' ', progressbar.Percentage(),
        ' ', progressbar.Bar(marker='❤'),
        ' ', progressbar.Timer(),
        ' ', progressbar.ETA(),
    ]

    
    with progressbar.ProgressBar(max_value=num_docs, widgets=widgets) as bar:
        for i, row in csv[0].iteritems():
            scored_docs.append(nlp(row))
            bar.update(i)

    df = pd.DataFrame(scored_docs)
    
    if dic_type == 'emfd':
        if prob_map == 'all' and out_metrics == 'sentiment':
            df['f_var'] = df[probabilites].var(axis=1)
            df['sent_var'] = df[senti].var(axis=1)
        elif prob_map == 'single' and out_metrics == 'sentiment':
            df['f_var'] = df[probabilites].var(axis=1)
            df['sent_var'] = df[senti].var(axis=1)
        elif prob_map == 'all' and out_metrics == 'vice-virtue':
            mfd_foundations = ['care.virtue', 'fairness.virtue', 'loyalty.virtue',
                   'authority.virtue','sanctity.virtue',
                   'care.vice','fairness.vice','loyalty.vice',
                   'authority.vice','sanctity.vice']
            df['f_var'] = df[mfd_foundations].var(axis=1)
            del df['moral']
        elif prob_map == 'single' and out_metrics == 'vice-virtue':
            mfd_foundations = ['care.virtue', 'fairness.virtue', 'loyalty.virtue',
                   'authority.virtue','sanctity.virtue',
                   'care.vice','fairness.vice','loyalty.vice',
                   'authority.vice','sanctity.vice']
            df['f_var'] = df[mfd_foundations].var(axis=1)
            
    if dic_type == 'mfd' or dic_type == 'mfd2':
        # Calculate variance
        mfd_foundations = ['care.virtue', 'fairness.virtue', 'loyalty.virtue',
                   'authority.virtue','sanctity.virtue',
                   'care.vice','fairness.vice','loyalty.vice',
                   'authority.vice','sanctity.vice']
        
        df['f_var'] = df[mfd_foundations].var(axis=1)
        
    return df

# PAT EXTRACTION #


def find_ent(token, entities):
    """High level function to match tokens to NER.
    Do not include in nlp.pipe!"""
    for k,v in entities.items():
        if token in v:
            return k

@Language.component("spacy_ner")
def spacy_ner(doc):
    include_ents = ['PERSON','NORP', 'GPE']
    entities = {ent.text:ent.text.split(' ') for ent in doc.ents if ent.label_ in include_ents}
    cc_processed = {e:{'patient_words':[], 'agent_words':[], 'attribute_words':[],
                  'patient_scores':[], 'agent_scores':[], 'attribute_scores':[]} for e in entities.keys()}
    ner_out = {'cc_processed':cc_processed, 'doc':doc, 'entities':entities}
    
    return ner_out

@Language.component("extract_dependencies")
def extract_dependencies(ner_out):
    doc = ner_out['doc']
    cc_processed= ner_out['cc_processed']
    entities = ner_out['entities']
    
    for token in doc:
        if token not in stopwords:
            if token.dep_ == 'nsubj' or  token.dep_ == 'ROOT':
                word = token.head.text.lower()
                if word in emfd.keys():
                    try:
                        cc_processed[find_ent(token.text, entities)]['agent_words'].append(word)
                        cc_processed[find_ent(token.text, entities)]['agent_scores'].append(emfd[word])
                    except KeyError as e:
                        pass

            if token.dep_ == 'dobj':
                word = token.head.text.lower()
                if word in emfd.keys():
                    try:
                        cc_processed[find_ent(token.text, entities)]['patient_words'].append(word)
                        cc_processed[find_ent(token.text, entities)]['patient_scores'].append(emfd[word])
                    except KeyError as e:
                        pass

            if token.dep_ == 'prep':
                word = token.head.text.lower()
                if word in emfd.keys():
                    for child in token.children:
                        try:
                            cc_processed[find_ent(str(child), entities)]['patient_words'].append(word)
                            cc_processed[find_ent(str(child), entities)]['patient_scores'].append(emfd[word])
                        except:
                            pass

            if token.text == 'is':
                try:
                    children = list(token.children)
                    word = children[1].lower()
                    if word in emfd.keys():
                        cc_processed[find_ent(str(children[0]),entities)]['attribute_words'].append(word)
                        cc_processed[find_ent(str(children[0]),entities)]['attribute_scores'].append(emfd[word])
                except:
                    pass

            if token.dep_ == 'attr':
                word = token.head.text.lower()
                if word in emfd.keys():
                    for child in token.children:
                        try:
                            cc_processed[find_ent(str(child), entities)]['attribute_words'].append(word)
                            cc_processed[find_ent(str(child), entities)]['attribute_scores'].append(emfd[word])
                        except:
                            pass   

            if token.dep_ == 'conj':
                if str(doc[token.right_edge.i]) == '.' or str(doc[token.right_edge.i]) == '!' or str(doc[token.right_edge.i]) == '?':
                    word = token.head.text.lower()
                    if word in emfd.keys():
                        try:
                            cc_processed[find_ent(str(doc[token.right_edge.i-1]), entities)]['agent_words'].append(word)
                            cc_processed[find_ent(str(doc[token.right_edge.i-1]), entities)]['agent_scores'].append(emfd[word])
                        except:
                            pass 
                else:
                    word = token.head.text.lower()
                    if word in emfd.keys():
                        try:
                            cc_processed[find_ent(str(token.right_edge), entities)]['agent_words'].append(word)
                            cc_processed[find_ent(str(token.right_edge), entities)]['agent_scores'].append(emfd[word])
                        except:
                            pass 
        
    return cc_processed

@Language.component("drop_ents")
def drop_ents(cc_processed):
    
    """Deletes entities w/out any related words."""
    
    empty_ents = []
    for k,v in cc_processed.items():
        counter = 0
        for k1, v1 in v.items():
            counter += len(v1)
        if counter == 0:
            empty_ents.append(k)
            
    for e in empty_ents:
        cc_processed.pop(e)
        
    return cc_processed

@Language.component("mean_pat")
def mean_pat(cc_processed):
    
    """Calculates the average emfd scores for
    words in each PAT category. 
    Returns the final dataframe for each document. 
    This frame has three columns for detected  words in each PAT category and
    10 columns for each PAT category capturing the mean emfd scores.
    """
    
    frames = []
    for k,v in cc_processed.items():
        agent = pd.DataFrame(v['agent_scores']).mean().to_frame().T
        agent.columns = ['agent_' + str(col) for col in agent.columns]
        
        patient = pd.DataFrame(v['patient_scores']).mean().to_frame().T
        patient.columns = ['patient_' + str(col) for col in patient.columns]
        
        attribute = pd.DataFrame(v['attribute_scores']).mean().to_frame().T
        attribute.columns = ['attribute_' + str(col) for col in attribute.columns]
        
        df = pd.concat([agent, patient, attribute], axis=1)
        df['NER'] = k
        df['agent_words'] = ','.join(v['agent_words'])
        df['patient_words'] = ','.join(v['patient_words'])
        df['attribute_words'] = ','.join((v['attribute_words']))
        frames.append(df)
    
    if len(frames) == 0:
        return pd.DataFrame()
    
    return pd.concat(frames)


def pat_docs(csv,num_docs):
    
    """Wrapper function that calls all individual functions
    to execute PAT extraction"""
    
    # Build spaCy pipeline
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("spacy_ner")
    nlp.add_pipe("extract_dependencies")
    nlp.add_pipe("drop_ents")
    nlp.add_pipe("mean_pat")
    
    scored_docs = []
    widgets = [
        'Processed: ', progressbar.Counter(),
        ' ', progressbar.Percentage(),
        ' ', progressbar.Bar(marker='❤'),
        ' ', progressbar.Timer(),
        ' ', progressbar.ETA(),
    ]
    
    with progressbar.ProgressBar(max_value=num_docs, widgets=widgets) as bar:
        for i, row in csv[0].iteritems():
            scored_docs.append(nlp(row))
            bar.update(i)
            
    df = pd.concat(scored_docs)
    
    words = ['agent_words','patient_words','attribute_words']
    a_mf = [c for c in df.columns if c.startswith('agent') and c.endswith('p')]
    a_sent = [c for c in df.columns if c.startswith('agent') and c.endswith('sent')]
    
    p_scores = [c for c in df.columns if c.startswith('patient') and c.endswith('p')]
    p_sent = [c for c in df.columns if c.startswith('patient') and c.endswith('sent')]
    
    att_scores = [c for c in df.columns if c.startswith('attribute') and c.endswith('p')]
    att_sent = [c for c in df.columns if c.startswith('attribute') and c.endswith('sent')]

    return df[['NER']+words+a_mf+a_sent+p_scores+p_sent+att_scores+att_sent].sort_values('NER')
