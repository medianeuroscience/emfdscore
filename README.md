## eMFDscore: Extended Moral Foundation Dictionary Scoring for Python 
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open Source Love png2](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

**eMFDscore** is a library for the fast and flexible extraction of various moral information metrics from textual input data. eMFDscore is built on [spaCy](https://github.com/explosion/spaCy) for faster execution and performs minimal preprocessing consisting of tokenization, syntactic dependency parsing, lower-casing, and stopword/punctuation/whitespace removal. eMFDscore lets users score documents with **multiple Moral Foundations Dictionaries**, provides **various metrics for analyzing moral information**, and extracts **moral patient, agent, and attribute words** related to entities.
    
When using eMFDscore, please consider giving this repository a star (top right corner) and citing the following article:  
Hopp, F. R., Fisher, J. T., Cornell, D., Huskey, R., & Weber, R. (2020). The extended Moral Foundations Dictionary (eMFD): Development and applications of a crowd-sourced approach to extracting moral intuitions from text. _Behavior Research Methods_, https://doi.org/10.3758/s13428-020-01433-0 

eMFDscore is dual-licensed under GNU GENERAL PUBLIC LICENSE 3.0, which permits the non-commercial use, distribution, and modification of the eMFDscore package. Commercial use of the eMFDscore requires an [application](https://forms.gle/RSKzZ2DvDyaprfeE8).

## Install 
**eMFDscore** requires a Python installation (v3.11+). If your machine does not have Python installed, we recommend installing Python by downloading and installing either [Anaconda or Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html) for your OS.

For best practises, we recommend installing eMFDscore into a virtual conda environment. Hence, you should first create a virtual environment by executing the following command in your terminal:

```
$ conda create -n emfd python=3.11
```

Once Anaconda/Miniconda is installed activate the env via:

```
$ source activate emfd
```

Next, you must install spaCy, which is the main natural language processing backend that emacscore is built on:

```
$ conda install -c conda-forge spacy
$ python -m spacy download en_core_web_sm
``` 

Finally, you can install eMFDscore by copying, pasting, and executing the following command: 

`
pip install https://github.com/medianeuroscience/emfdscore/archive/master.zip
`

### eMFDscore in Google Colaboratory

eMFDscore can also be run in [google colab](https://colab.research.google.com/notebooks/intro.ipynb). All you need to do is add these lines to the beginning of your notebook, execute them, and then restart your runtime:

```
!pip install https://github.com/medianeuroscience/emfdscore/archive/master.zip
```

You can then use eMFDscore as a regular python library.
