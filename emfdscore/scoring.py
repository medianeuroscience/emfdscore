import nltk
from nltk.corpus import stopwords as nltk_sw

from sklearn.feature_extraction import text
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_sw
from spacy.language import Language
from spacy.tokens import Doc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from collections import Counter
from emfdscore.load_mfds import *
import progressbar

try:
    nltk_stopwords = nltk_sw.words('english')
except Exception:
    print('NLTK stopwords missing, downloading now.')
    nltk.download('stopwords')
    nltk_stopwords = nltk_sw.words('english')

STOPWORDS_SET = set(list(nltk_stopwords) + list(text.ENGLISH_STOP_WORDS) + list(spacy_sw))

# -----------------------------
# spaCy Doc extensions
# -----------------------------
def _ensure_doc_extensions():
    # token cache
    if not Doc.has_extension("mfd_tokens"):
        Doc.set_extension("mfd_tokens", default=None)

    # scoring outputs
    for ext in [
        "score_emfd_all_sent",
        "score_emfd_single_sent",
        "score_emfd_all_vice_virtue",
        "score_emfd_single_vice_virtue",
        "score_mfd",
        "score_mfd2",
    ]:
        if not Doc.has_extension(ext):
            Doc.set_extension(ext, default={})

    # PAT intermediates/outputs
    if not Doc.has_extension("pat_entities"):
        Doc.set_extension("pat_entities", default=None)
    if not Doc.has_extension("pat_cc_processed"):
        Doc.set_extension("pat_cc_processed", default=None)
    if not Doc.has_extension("pat_df"):
        Doc.set_extension("pat_df", default=None)

_ensure_doc_extensions()


def _safe_ratio(num: int, denom: int) -> float:
    return num / (denom if denom != 0 else 1)

def _first_text_column(df: pd.DataFrame) -> pd.Series:
    # Return the first column as text series (compatible with prior CSV format)
    return df.iloc[:, 0]

# -----------------------------
# Tokenizer (spaCy-compliant)
# -----------------------------
@Language.component("mfd_tokenizer")
def mfd_tokenizer(doc):
    filtered_tokens = [
        t for t in doc
        if t.lower_ not in STOPWORDS_SET
        and not t.is_punct
        and not t.is_digit
        and not t.is_quote
        and not t.like_num
        and not t.is_space
    ]
    doc._.mfd_tokens = [t.lower_ for t in filtered_tokens]
    return doc


@Language.component("score_emfd_all_sent")
def score_emfd_all_sent(doc):
    tokens = doc._.mfd_tokens or [t.lower_ for t in doc]
    emfd_score = {k: 0 for k in probabilites + senti}
    moral_words = [emfd[t] for t in tokens if t in emfd]
    for dic in moral_words:
        for k in probabilites + senti:
            emfd_score[k] += dic[k]
    if moral_words:
        emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
        emfd_score['moral_nonmoral_ratio'] = _safe_ratio(len(moral_words), len(tokens)-len(moral_words))
    else:
        emfd_score = {k: 0 for k in probabilites + senti}
        emfd_score['moral_nonmoral_ratio'] = 0
    doc._.score_emfd_all_sent = emfd_score
    return doc

@Language.component("score_emfd_single_sent")
def score_emfd_single_sent(doc):
    tokens = doc._.mfd_tokens or [t.lower_ for t in doc]
    emfd_score = {k: 0 for k in probabilites + senti}
    moral_words = [emfd_single_sent[t] for t in tokens if t in emfd_single_sent]
    for dic in moral_words:
        base_f = dic['foundation'].split('_')[0]
        emfd_score[dic['foundation']] += dic['score']
        emfd_score[base_f + '_sent'] += dic['sentiment']
    if moral_words:
        emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
        emfd_score['moral_nonmoral_ratio'] = _safe_ratio(len(moral_words), len(tokens)-len(moral_words))
    else:
        emfd_score = {k: 0 for k in probabilites + senti}
        emfd_score['moral_nonmoral_ratio'] = 0
    doc._.score_emfd_single_sent = emfd_score
    return doc

@Language.component("score_emfd_all_vice_virtue")
def score_emfd_all_vice_virtue(doc):
    tokens = doc._.mfd_tokens or [t.lower_ for t in doc]
    emfd_score = {k: 0 for k in mfd_foundations}
    moral_words = [emfd_all_vice_virtue[t] for t in tokens if t in emfd_all_vice_virtue]
    for dic in moral_words:
        for f in mfd_foundations:
            if f != 'moral':
                emfd_score[f] += dic[f]
    if moral_words:
        emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
        emfd_score['moral_nonmoral_ratio'] = _safe_ratio(len(moral_words), len(tokens)-len(moral_words))
    else:
        emfd_score = {k: 0 for k in mfd_foundations}
        emfd_score['moral_nonmoral_ratio'] = 0
    doc._.score_emfd_all_vice_virtue = emfd_score
    return doc

@Language.component("score_emfd_single_vice_virtue")
def score_emfd_single_vice_virtue(doc):
    tokens = doc._.mfd_tokens or [t.lower_ for t in doc]
    emfd_score = {k: 0 for k in mfd_foundations if k != 'moral'}
    moral_words = [emfd_single_vice_virtue[t] for t in tokens if t in emfd_single_vice_virtue]
    for dic in moral_words:
        emfd_score[dic['foundation']] += dic['score']
    if moral_words:
        emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
        emfd_score['moral_nonmoral_ratio'] = _safe_ratio(len(moral_words), len(tokens)-len(moral_words))
    else:
        emfd_score = {k: 0 for k in mfd_foundations if k != 'moral'}
        emfd_score['moral_nonmoral_ratio'] = 0
    doc._.score_emfd_single_vice_virtue = emfd_score
    return doc

@Language.component("score_mfd")
def score_mfd(doc):
    tokens = doc._.mfd_tokens or [t.lower_ for t in doc]
    mfd_score = {k: 0 for k in mfd_foundations}
    moral_words = 0
    for tok in tokens:
        for v in mfd_regex.keys():
            if mfd_regex[v].match(tok):
                moral_words += 1
                for f in mfd[v]:
                    mfd_score[f] += 1
    if moral_words:
        mfd_score = {k: v/moral_words for k, v in mfd_score.items()}
        mfd_score['moral_nonmoral_ratio'] = _safe_ratio(moral_words, len(tokens)-moral_words)
    else:
        mfd_score = {k: 0 for k in mfd_foundations}
        mfd_score['moral_nonmoral_ratio'] = 0
    doc._.score_mfd = mfd_score
    return doc

@Language.component("score_mfd2")
def score_mfd2(doc):
    tokens = doc._.mfd_tokens or [t.lower_ for t in doc]
    mfd2_score = {k: 0 for k in mfd2_foundations}
    moral_found = [mfd2[t]['foundation'] for t in tokens if t in mfd2]
    f_counts = Counter(moral_found)
    mfd2_score.update(f_counts)
    if moral_found:
        mfd2_score = {k: v/len(moral_found) for k, v in mfd2_score.items()}
        mfd2_score['moral_nonmoral_ratio'] = _safe_ratio(len(moral_found), len(tokens)-len(moral_found))
    else:
        mfd2_score = {k: 0 for k in mfd2_foundations}
        mfd2_score['moral_nonmoral_ratio'] = 0
    doc._.score_mfd2 = mfd2_score
    return doc


def score_docs(csv, dic_type, prob_map, score_type, out_metrics, num_docs):

    if score_type == 'wordlist':
        widgets = ['Processed: ', progressbar.Counter(), ' ', progressbar.Percentage(),
                   ' ', progressbar.Bar(marker='❤'), ' ', progressbar.Timer(), ' ', progressbar.ETA()]
        with progressbar.ProgressBar(max_value=num_docs, widgets=widgets) as bar:
            moral_words = []
            # Expect the first (only) column to contain words
            for i, row in _first_text_column(csv).items():
                if row in emfd:
                    moral_words.append(emfd[row])
                bar.update(i if isinstance(i, int) else 0)

            emfd_score = {k: 0 for k in probabilites + senti}
            for dic in moral_words:
                for k in probabilites + senti:
                    emfd_score[k] += dic[k]

            if moral_words:
                emfd_score = {k: v/len(moral_words) for k, v in emfd_score.items()}
            else:
                emfd_score = {k: 0 for k in probabilites + senti}

            emfd_score['cnt'] = len(moral_words)
            df = pd.DataFrame(pd.Series(emfd_score)).T
            df = df[['cnt'] + probabilites + senti]
            return df

    if score_type == 'gdelt.ngrams':
        widgets = ['Processed: ', progressbar.Counter(), ' ', progressbar.Percentage(),
                   ' ', progressbar.Bar(marker='❤'), ' ', progressbar.Timer(), ' ', progressbar.ETA()]
        with progressbar.ProgressBar(max_value=num_docs, widgets=widgets) as bar:
            moral_words = []
            freqs = []
            # Expect dataframe with columns: 'word', 'freq'
            for i, row in csv.iterrows():
                w = row['word']
                f = int(row['freq'])
                if w in emfd:
                    moral_words.append({'scores': emfd[w], 'freq': f})
                    freqs.append(f)
                bar.update(i if isinstance(i, int) else 0)

            emfd_score = {k: 0 for k in probabilites + senti}
            for dic in moral_words:
                for k in probabilites + senti:
                    emfd_score[k] += dic['scores'][k] * dic['freq']

            total = sum(freqs) if freqs else 1
            emfd_score = {k: v/total for k, v in emfd_score.items()}
            emfd_score['cnt'] = total
            df = pd.DataFrame(pd.Series(emfd_score)).T
            df = df[['cnt'] + probabilites + senti]
            return df

    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    nlp.add_pipe("mfd_tokenizer")

    if dic_type == 'emfd':
        if prob_map == 'all' and out_metrics == 'sentiment':
            nlp.add_pipe("score_emfd_all_sent", last=True)
        elif prob_map == 'single' and out_metrics == 'sentiment':
            nlp.add_pipe("score_emfd_single_sent", last=True)
        elif prob_map == 'all' and out_metrics == 'vice-virtue':
            nlp.add_pipe("score_emfd_all_vice_virtue", last=True)
        elif prob_map == 'single' and out_metrics == 'vice-virtue':
            nlp.add_pipe("score_emfd_single_vice_virtue", last=True)
        else:
            print("Invalid emfd configuration for prob_map/out_metrics")
            return
    elif dic_type == 'mfd':
        nlp.add_pipe("score_mfd", last=True)
    elif dic_type == 'mfd2':
        nlp.add_pipe("score_mfd2", last=True)
    else:
        print('Dictionary type not recognized. Available values are: emfd, mfd, mfd2')
        return

    scored_docs = []
    widgets = ['Processed: ', progressbar.Counter(), ' ', progressbar.Percentage(),
               ' ', progressbar.Bar(marker="❤"), ' ', progressbar.Timer(), ' ', progressbar.ETA()]

    with progressbar.ProgressBar(max_value=num_docs, widgets=widgets) as bar:
        for i, (_, row) in enumerate(_first_text_column(csv).items(), start=1):
            doc = nlp(row)
            if dic_type == 'emfd':
                if prob_map == 'all' and out_metrics == 'sentiment':
                    scored_docs.append(doc._.score_emfd_all_sent)
                elif prob_map == 'single' and out_metrics == 'sentiment':
                    scored_docs.append(doc._.score_emfd_single_sent)
                elif prob_map == 'all' and out_metrics == 'vice-virtue':
                    scored_docs.append(doc._.score_emfd_all_vice_virtue)
                elif prob_map == 'single' and out_metrics == 'vice-virtue':
                    scored_docs.append(doc._.score_emfd_single_vice_virtue)
            elif dic_type == 'mfd':
                scored_docs.append(doc._.score_mfd)
            elif dic_type == 'mfd2':
                scored_docs.append(doc._.score_mfd2)
            bar.update(i)

    df = pd.DataFrame(scored_docs)

    # Retain original variance calculations
    if dic_type == 'emfd':
        if out_metrics == 'sentiment':  # both 'all' and 'single'
            if all(c in df.columns for c in probabilites):
                df['f_var'] = df[probabilites].var(axis=1)
            if all(c in df.columns for c in senti):
                df['sent_var'] = df[senti].var(axis=1)
        elif out_metrics == 'vice-virtue':
            vv_cols = ['care.virtue', 'fairness.virtue', 'loyalty.virtue',
                       'authority.virtue', 'sanctity.virtue',
                       'care.vice', 'fairness.vice', 'loyalty.vice',
                       'authority.vice', 'sanctity.vice']
            vv_present = [c for c in vv_cols if c in df.columns]
            if vv_present:
                df['f_var'] = df[vv_present].var(axis=1)
            if 'moral' in df.columns:
                del df['moral']
    elif dic_type in ['mfd', 'mfd2']:
        vv_cols = ['care.virtue', 'fairness.virtue', 'loyalty.virtue',
                   'authority.virtue', 'sanctity.virtue',
                   'care.vice', 'fairness.vice', 'loyalty.vice',
                   'authority.vice', 'sanctity.vice']
        vv_present = [c for c in vv_cols if c in df.columns]
        if vv_present:
            df['f_var'] = df[vv_present].var(axis=1)

    return df

def find_ent(token, entities):
    for k, v in entities.items():
        if token in v:
            return k
    return None

@Language.component("spacy_ner")
def spacy_ner(doc):
    include_ents = ['PERSON', 'NORP', 'GPE']
    entities = {ent.text: ent.text.split(' ') for ent in doc.ents if ent.label_ in include_ents}
    cc_processed = {
        e: {
            'patient_words': [], 'agent_words': [], 'attribute_words': [],
            'patient_scores': [], 'agent_scores': [], 'attribute_scores': []
        } for e in entities.keys()
    }
    doc._.pat_entities = entities
    doc._.pat_cc_processed = cc_processed
    return doc

@Language.component("extract_dependencies")
def extract_dependencies(doc):
    cc_processed = doc._.pat_cc_processed or {}
    entities = doc._.pat_entities or {}

    for token in doc:
        tok_lower = token.text.lower()
        if tok_lower in STOPWORDS_SET:
            continue

        if token.dep_ in {'nsubj', 'ROOT'}:
            word = token.head.text.lower()
            if word in emfd:
                ent = find_ent(token.text, entities)
                if ent is not None:
                    cc_processed[ent]['agent_words'].append(word)
                    cc_processed[ent]['agent_scores'].append(emfd[word])

        if token.dep_ == 'dobj':
            word = token.head.text.lower()
            if word in emfd:
                ent = find_ent(token.text, entities)
                if ent is not None:
                    cc_processed[ent]['patient_words'].append(word)
                    cc_processed[ent]['patient_scores'].append(emfd[word])

        if token.dep_ == 'prep':
            word = token.head.text.lower()
            if word in emfd:
                for child in token.children:
                    ent = find_ent(str(child), entities)
                    if ent is not None:
                        cc_processed[ent]['patient_words'].append(word)
                        cc_processed[ent]['patient_scores'].append(emfd[word])

        if token.text == 'is':
            children = list(token.children)
            if len(children) >= 2:
                word = children[1].text.lower()
                subj = children[0].text
                if word in emfd:
                    ent = find_ent(subj, entities)
                    if ent is not None:
                        cc_processed[ent]['attribute_words'].append(word)
                        cc_processed[ent]['attribute_scores'].append(emfd[word])

        if token.dep_ == 'attr':
            word = token.head.text.lower()
            if word in emfd:
                for child in token.children:
                    ent = find_ent(str(child), entities)
                    if ent is not None:
                        cc_processed[ent]['attribute_words'].append(word)
                        cc_processed[ent]['attribute_scores'].append(emfd[word])

        if token.dep_ == 'conj':
            word = token.head.text.lower()
            if word in emfd:
                if str(doc[token.right_edge.i]) in {'.', '!', '?'}:
                    anchor = str(doc[token.right_edge.i - 1])
                else:
                    anchor = str(token.right_edge)
                ent = find_ent(anchor, entities)
                if ent is not None:
                    cc_processed[ent]['agent_words'].append(word)
                    cc_processed[ent]['agent_scores'].append(emfd[word])

    doc._.pat_cc_processed = cc_processed
    return doc

@Language.component("drop_ents")
def drop_ents(doc):
    cc_processed = doc._.pat_cc_processed or {}
    empty = []
    for k, v in cc_processed.items():
        total = 0
        for v1 in v.values():
            total += len(v1)
        if total == 0:
            empty.append(k)
    for e in empty:
        cc_processed.pop(e, None)
    doc._.pat_cc_processed = cc_processed
    return doc

@Language.component("mean_pat")
def mean_pat(doc):
    cc_processed = doc._.pat_cc_processed or {}
    frames = []
    for k, v in cc_processed.items():
        agent = pd.DataFrame(v['agent_scores']).mean().to_frame().T if v['agent_scores'] else pd.DataFrame()
        if not agent.empty:
            agent.columns = ['agent_' + str(col) for col in agent.columns]
        patient = pd.DataFrame(v['patient_scores']).mean().to_frame().T if v['patient_scores'] else pd.DataFrame()
        if not patient.empty:
            patient.columns = ['patient_' + str(col) for col in patient.columns]
        attribute = pd.DataFrame(v['attribute_scores']).mean().to_frame().T if v['attribute_scores'] else pd.DataFrame()
        if not attribute.empty:
            attribute.columns = ['attribute_' + str(col) for col in attribute.columns]
        if agent.empty and patient.empty and attribute.empty:
            continue
        df = pd.concat([agent, patient, attribute], axis=1)
        df['NER'] = k
        df['agent_words'] = ','.join(v['agent_words'])
        df['patient_words'] = ','.join(v['patient_words'])
        df['attribute_words'] = ','.join(v['attribute_words'])
        frames.append(df)
    doc._.pat_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return doc

def pat_docs(csv, num_docs):
    
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("spacy_ner")
    nlp.add_pipe("extract_dependencies")
    nlp.add_pipe("drop_ents")
    nlp.add_pipe("mean_pat")

    dfs = []
    widgets = ['Processed: ', progressbar.Counter(), ' ', progressbar.Percentage(),
               ' ', progressbar.Bar(marker="❤"), ' ', progressbar.Timer(), ' ', progressbar.ETA()]
    with progressbar.ProgressBar(max_value=num_docs, widgets=widgets) as bar:
        for i, (_, text_) in enumerate(_first_text_column(csv).items(), start=1):
            doc = nlp(text_)
            if isinstance(doc._.pat_df, pd.DataFrame) and not doc._.pat_df.empty:
                dfs.append(doc._.pat_df)
            bar.update(i)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    words = ['agent_words', 'patient_words', 'attribute_words']
    a_mf = [c for c in df.columns if c.startswith('agent') and c.endswith('p')]
    a_sent = [c for c in df.columns if c.startswith('agent') and c.endswith('sent')]
    p_scores = [c for c in df.columns if c.startswith('patient') and c.endswith('p')]
    p_sent = [c for c in df.columns if c.startswith('patient') and c.endswith('sent')]
    att_scores = [c for c in df.columns if c.startswith('attribute') and c.endswith('p')]
    att_sent = [c for c in df.columns if c.startswith('attribute') and c.endswith('sent')]

    cols = ['NER'] + words + a_mf + a_sent + p_scores + p_sent + att_scores + att_sent
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values('NER') if 'NER' in df.columns else df[cols]
