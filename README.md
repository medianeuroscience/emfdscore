# ❤ AMorE: Automated Morality Extraction for Python

The Automated Morality Extractor (AMorE) is a Python library for the fast and flexible extraction of various moral information metrics from textual input data. AMoRe is build on [spaCy](https://github.com/explosion/spaCy) for faster execution and performs minimal preprocessing consisting of tokenziation, syntactic dependency parsing, lower-casing, and stopword/punctuation/whitespace removal. AMorE lets users employ and compare **multiple Moral Foundations Dictionaries**, provides **various metrics for analyzing moral information**, and extracts **moral patient, agent, and attribute words** related to entities.
    
When using AMorE, please consider citing the following article: redacted for anonymous review. 

AMorE can be installed via pip: `pip install py-amore`

## Install 
AMorE requires a Python installation (v3.5+). If your machine does not have Python installed, we recommend installing Python by downloading and installing either [Anaconda or Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html) for your OS. 

Once Anaconda/Miniconda is installed, you can install AMorE by opening a terminal and typing: `pip install amore`

## Usage 
AMorE is executed via the command line (terminal). 
A typical command specifies the following: 

`$amore [INPUT.CSV] [DICT_TYPE] [SCORING_METHOD] [OUTPUT.CSV]`

- INPUT_TYPE = The path to a CSV in which the first column contains the document texts to be scored. Each row should reflect its own document. See the template_input.csv for an example file. 

- DICT_TYPE = Declares which MFD is applied to score documents. In its current version, AMorE lets users choose between three dictionaries: `emfd` = extended Moral Foundations Dictionary (e-MFD; under review); `mfd2` = Moral Foundations Dicitonary 2.0 (Frimer et al., 2017; https://osf.io/xakyw/ ); `mfd` = original Moral Foundations Dictionary (https://moralfoundations.org/othermaterials) 

- SCORING_METHOD = Currently, AMorE employs two different scoring algorithms: `bow` is a classical Bag-of-Words approach in which the algorithm simply searches for word matches between document texts and the specified dictionary. `pat` relies on named entity recognition and syntactic dependency parsing. For each document, the algorithm first extracts all mentioned entities. Next, for each entitiy, AMorE extracts words that pertain to 1) moral verbs for which the entity is an agent argument (Agent verbs), 2) moral verbs for which the entity is the patient, theme, or other argument (Patient verbs), and other moral attributes (i.e., adjectival modifiers, appositives, etc.). **Note**: Only the emfd is currenlty supported for PAT extraction.

- OUTPUT.csv = Specifies the file name of the generated output csv. 

Click on the below terminak for a usage demonstration. 
[![asciicast](https://asciinema.org/a/HcnMC8fyBqZD3BdTG0fxZNsNh.svg)](https://asciinema.org/a/HcnMC8fyBqZD3BdTG0fxZNsNh?autoplay=1&theme=solarized-dark)

## Returned Metrics
TODO. 
