<img src=https://github.com/medianeuroscience/pyamore/blob/master/pyamore_logo.png width="100" height="50">

## Automated Morality Extraction for Python
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/) [![Open Source Love png2](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

The Automated Morality Extraction for Python (PyAMorE) is a Python library for the fast and flexible extraction of various moral information metrics from textual input data. AMoRe is build on [spaCy](https://github.com/explosion/spaCy) for faster execution and performs minimal preprocessing consisting of tokenization, syntactic dependency parsing, lower-casing, and stopword/punctuation/whitespace removal. Py-AMorE lets users employ and compare **multiple Moral Foundations Dictionaries**, provides **various metrics for analyzing moral information**, and extracts **moral patient, agent, and attribute words** related to entities.
    
When using PyAMorE, please consider citing the following article: _The extended Moral Foundations Dictionary (e-MFD): Development and applications of a crowd-sourced moral foundations dictionary._

## Install 
PyAMorE requires a Python installation (v3.5+). If your machine does not have Python installed, we recommend installing Python by downloading and installing either [Anaconda or Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html) for your OS.

For best practises, we recommend installing PyAMorE into a virtual conda environment:
`
conda create -n yourenvname python=3.5
`

`
source activate yourenvname
`

Once Anaconda/Miniconda is installed and the env activated, you can install PyAMorE by copying, pasting, and executing the following command: 

`
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyamore
`

**NOTE** The install command will be simplified after publication. 

After installing PyAMorE, please install the latest version of spaCy by running the following two commands in your terminal:

`
pip install -U spacy
`

`
python -m spacy download en
`

## Usage 
PyAMorE is executed via the command line (terminal). 
A typical command specifies the following: 

`$ pyamore [INPUT.CSV] [DICT_TYPE] [SCORING_METHOD] [OUTPUT.CSV]`

- INPUT.CSV = The path to a CSV in which the first column contains the document texts to be scored. Each row should reflect its own document. See the [template_input.csv](https://github.com/medianeuroscience/pyamore/blob/master/pyamore/template_input.csv) for an example file format.

- DICT_TYPE = Declares which MFD is applied to score documents. In its current version, AMorE lets users choose between three dictionaries: `emfd` = extended Moral Foundations Dictionary (e-MFD; under review); `mfd2` = Moral Foundations Dicitonary 2.0 (Frimer et al., 2017; https://osf.io/xakyw/ ); `mfd` = original Moral Foundations Dictionary (https://moralfoundations.org/othermaterials) 

- SCORING_METHOD = Currently, PyAMorE employs two different scoring algorithms: `bow` is a classical Bag-of-Words approach in which the algorithm simply searches for word matches between document texts and the specified dictionary. `pat` relies on named entity recognition and syntactic dependency parsing. For each document, the algorithm first extracts all mentioned entities. Next, for each entitiy, PyAMorE extracts words that pertain to 1) moral verbs for which the entity is an agent argument (Agent verbs), 2) moral verbs for which the entity is the patient, theme, or other argument (Patient verbs), and other moral attributes (i.e., adjectival modifiers, appositives, etc.). **Note**: Only the emfd is currenlty supported for PAT extraction. Additionally, this method is more computationally expensive and thus has a longer execution time. 

- OUTPUT.csv = Specifies the file name of the generated output csv. 

Click on the below terminal for a usage demonstration.
[![asciicast](https://asciinema.org/a/pE2VgwtS8Z3A2uUIZcuayLVWq.svg)](https://asciinema.org/a/pE2VgwtS8Z3A2uUIZcuayLVWq?autoplay=1&theme=solarized-dark)

## Returned Metrics
**BoW Scoring**

Slightly different output metrics are provided depending on the dictionary that was used for scoring. Regardless of dictionary type, each row in the output.csv contains the produced metrics for the respective document in the input.csv

_e-MFD_: 
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
