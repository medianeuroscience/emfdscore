import re, fnmatch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas import DataFrame, read_pickle,read_csv
import os


# Define path for module to allow dict imports. 
fileDir = os.path.dirname(os.path.abspath(__file__))


# Load E-MFD
emfd = read_csv(fileDir+'/dictionaries/emfd_scoring.csv',index_col='word')
probabilites = [c for c in emfd.columns if c.endswith('_p')]
foundations = ['care','fairness','loyalty','authority','sanctity']
senti = [c for c in emfd.columns if c.endswith('_sent')]
emfd = emfd.T.to_dict()

# Load eMFD-all-vice-virtue
emfd_all_vice_virtue = read_pickle(fileDir+'/dictionaries/emfd_all_vice_virtue.pkl')

# Load eMFD-single-vice-virtue
emfd_single_vice_virtue = read_pickle(fileDir+'/dictionaries/emfd_single_vice_virtue.pkl')

# Load eMFD-single-sent
emfd_single_sent = read_pickle(fileDir+'/dictionaries/emfd_single_sent.pkl')

# Load MFD
MFD = fileDir+'/dictionaries/mft_original.dic'
nummap = dict()
mfd = dict()
mfd_regex = dict()
wordmode = True
with open(MFD, 'r') as f:
    for line in f.readlines():
        ent = line.strip().split()
        if line[0] == '%':
            wordmode = not wordmode
        elif len(ent) > 0:
            if wordmode:
                mfd[ent[0]] = [nummap[e] for e in ent[1:]]
            else:
                nummap[ent[0]] = ent[1]
                
mfd_foundations = ['care.virtue', 'fairness.virtue', 'loyalty.virtue',
                   'authority.virtue','sanctity.virtue',
                   'care.vice','fairness.vice','loyalty.vice',
                   'authority.vice','sanctity.vice','moral']

for v in mfd.keys():
    mfd_regex[v] = re.compile(fnmatch.translate(v))
    
# Load MFD2.0 
MFD2 = fileDir+'/dictionaries/mfd2.0.dic'
nummap = dict()
mfd2 = dict()
wordmode = True
with open(MFD2, 'r') as f:
    for line in f.readlines():
        ent = line.strip().split()
        if line[0] == '%':
            wordmode = not wordmode
        elif len(ent) > 0:
            if wordmode:
                wordkey = ''.join([e for e in ent if e not in nummap.keys()])
                mfd2[wordkey] = [nummap[e] for e in ent if e in nummap.keys()]
            else:
                nummap[ent[0]] = ent[1]

mfd2 = DataFrame.from_dict(mfd2).T
mfd2_foundations = mfd2[0].unique()
mfd2['foundation'] = mfd2[0]
del mfd2[0]
mfd2 = mfd2.T.to_dict()
