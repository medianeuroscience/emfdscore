## eMFDscore: Extended Moral Foundation Dictionary Scoring for Python 
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/) [![Open Source Love png2](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

**eMFDscore** is a library for the fast and flexible extraction of various moral information metrics from textual input data. eMFDscore is build on [spaCy](https://github.com/explosion/spaCy) for faster execution and performs minimal preprocessing consisting of tokenization, syntactic dependency parsing, lower-casing, and stopword/punctuation/whitespace removal. eMFDscore lets users score documents with **multiple Moral Foundations Dictionaries**, provides **various metrics for analyzing moral information**, and extracts **moral patient, agent, and attribute words** related to entities.
    
When using eMFDscore, please consider giving this repository a star (top right corner) and citing the following article: _The Extended Moral Foundations Dictionary (eMFD): Development and Applications of a Crowd-Sourced Approach to Extracting Moral Intuitions from Text_

## Install 
eMFDscore requires a Python installation (v3.5+). If your machine does not have Python installed, we recommend installing Python by downloading and installing either [Anaconda or Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html) for your OS.

For best practises, we recommend installing eMFDscore into a virtual conda environment. Hence, you should first create a virtual environment by executing the following command in your terminal:

```
$ conda create -n yourenvname python=3.7
```

Once Anaconda/Miniconda is installed activate the env via:

```
$ source activate yourenvname
```

Next, you must install spaCy, which is the main natural language processing backend that eMFDscore is built on:

```
$ conda install -c conda-forge spacy
$ python -m spacy download en_core_web_sm
``` 

Finally, you can install eMFDscore by copying, pasting, and executing the following command: 

`
pip install https://github.com/medianeuroscience/emfdscore/archive/master.zip
`


## Usage 
eMFDscore is executed via the command line (terminal). 
A typical command specifies the following: 

`$ emfdscore [INPUT.CSV] [DICT_TYPE] [SCORING_METHOD] [OUTPUT.CSV]`

- INPUT.CSV = The path to a CSV in which the first column contains the document texts to be scored. Each row should reflect its own document. See the [template_input.csv](https://github.com/medianeuroscience/emfdscore/blob/master/emfdscore/template_input.csv) for an example file format.

- DICT_TYPE = Declares which MFD is applied to score documents. In its current version, eMFDscore lets users choose between three dictionaries: `emfd` = extended Moral Foundations Dictionary (eMFD; under review); `mfd2` = Moral Foundations Dicitonary 2.0 (Frimer et al., 2017; https://osf.io/xakyw/ ); `mfd` = original Moral Foundations Dictionary (https://moralfoundations.org/othermaterials) 

- SCORING_METHOD = Currently, eMFDscore employs three different scoring algorithms:   

    - `bow` is a classical Bag-of-Words approach in which the algorithm simply searches for word matches between document texts and the specified dictionary.  

    - `pat` relies on named entity recognition and syntactic dependency parsing. For each document, the algorithm first extracts all mentioned entities. Next, for each entitiy, eMFDscore extracts words that pertain to 1) moral verbs for which the entity is an agent argument (Agent verbs), 2) moral verbs for which the entity is the patient, theme, or other argument (Patient verbs), and other moral attributes (i.e., adjectival modifiers, appositives, etc.).  

    - `wordlist` is a simple scoring algorithm that lets users examine the moral content of individual words. This scoring method expects a CSV where each row corresponds to a unique word. **Note**: The `wordlist` scoring algorithm does not perform any tokenization or preprocessing on the wordlists. For a more fine-grained moral content extraction, users are encouraged to use either the `bow` or `path` methodology. Furthermore, only the emfd is currenlty supported for PAT extraction. Additionally, this method is more computationally expensive and thus has a longer execution time. 
    
    - `gdelt.ngrams` is designed for the [Global Database of Events, Language, and Tone](https://blog.gdeltproject.org/announcing-the-television-news-ngram-datasets-tv-ngram/) Television Ngram dataset. This scoring method expects a unigram (1gram) input text file from GDELT and will score each unprocessed (untokenized) unigram with the eMFD. 

- OUTPUT.csv = Specifies the file name of the generated output csv. 

Click on the below terminal for a usage demonstration.
[![asciicast](https://asciinema.org/a/95Vr51C90rcXHeFkLLfCUOgKH.svg)](https://asciinema.org/a/95Vr51C90rcXHeFkLLfCUOgKH?autoplay=1&theme=solarized-dark&speed=2)

## Returned Metrics
**BoW Scoring**

Slightly different output metrics are provided depending on the dictionary that was used for scoring. Regardless of dictionary type, each row in the output.csv contains the produced metrics for the respective document in the input.csv

_eMFD_: 
- Five scores that denote the average presence of each moral foundation (columns ending with _p_) 
- Five scores that denote the upholding (positive values) or violation (negative values) of each moral foundation (columns ending with _sent_) 
- The variance across the five moral foundation scores 
- The variance across the five sentiment scores
- A ratio of detected moral words to non-moral words

_MFD2.0_ & _MFD_:
- Ten columns that denote the average presence of each moral foundation category. 
- The variance across the ten moral foundation scores 
- A ratio of detected moral words to non-moral words

**PAT Scoring**

When choosing the pat option for document scoring, the returned CSV is sorted and indexed by each mentioned entity across input documents. Note that duplicate entity mentions are possible and deliberate. For each entity, the following variables are provided:

- agent_words: Moral words for which the entity was declared the actor/agent 
- patient_words: Moral words for which the entity was declared the patient/target
- attribute_words: Moral words used to describe the entity (usually very few detected)
- For each of the above categories, ten variables are computed. The first five specify the average presence of each moral foundation across the detected words (columns ending with _p_). The second five specify the average upholding (positive values)/violation(negative values) of each moral foundation (columns ending with _sent_). 
