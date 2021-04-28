## eMFDscore: Extended Moral Foundation Dictionary Scoring for Python 
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open Source Love png2](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

**eMFDscore** is a library for the fast and flexible extraction of various moral information metrics from textual input data. eMFDscore is build on [spaCy](https://github.com/explosion/spaCy) for faster execution and performs minimal preprocessing consisting of tokenization, syntactic dependency parsing, lower-casing, and stopword/punctuation/whitespace removal. eMFDscore lets users score documents with **multiple Moral Foundations Dictionaries**, provides **various metrics for analyzing moral information**, and extracts **moral patient, agent, and attribute words** related to entities.
    
When using eMFDscore, please consider giving this repository a star (top right corner) and citing the following article:  
Hopp, F. R., Fisher, J. T., Cornell, D., Huskey, R., & Weber, R. (2020). The extended Moral Foundations Dictionary (eMFD): Development and applications of a crowd-sourced approach to extracting moral intuitions from text. _Behavior Research Methods_, https://doi.org/10.3758/s13428-020-01433-0 

eMFDscore is dual-licensed under GNU GENERAL PUBLIC LICENSE 3.0, which permits the non-commercial use, distribution, and modification of the eMFDscore package. Commercial use of the eMFDscore requires an [application](https://forms.gle/RSKzZ2DvDyaprfeE8).

## Install 
eMFDscore requires a Python installation (v3.7+). If your machine does not have Python installed, we recommend installing Python by downloading and installing either [Anaconda or Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html) for your OS.

For best practises, we recommend installing eMFDscore into a virtual conda environment. Hence, you should first create a virtual environment by executing the following command in your terminal:

```
$ conda create -n emfd python=3.7
```

Once Anaconda/Miniconda is installed activate the env via:

```
$ source activate emfd
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
Please refer to this [tutorial](https://github.com/medianeuroscience/emfdscore/blob/master/eMFDscore_Tutorial.ipynb) to learn how to use eMFDscore. 

If you are using the eMFD within the [Global Database of Events, Language, and Tone (GDELT)](https://blog.gdeltproject.org/examining-trends-in-moral-news-framing-across-a-decade-of-television-coverage/) please read the following [documentation](https://github.com/medianeuroscience/emfdscore/blob/master/emfd_gdelt_readme.pdf).

## Applications 
The eMFD has been used in the following applications:
- [Priniski, J. H., Mokhberian, N., Harandizadeh, B., Morstatter, F., Lerman, K., Lu, H., & Brantingham, P. J. (2021). Mapping Moral Valence of Tweets Following the Killing of George Floyd. arXiv preprint arXiv:2104.09578.](https://arxiv.org/abs/2104.09578)

For using the eMFD on shorter texts (e.g., tweets and news headlines), we suggest to apply the eMFD in a [FrameAxis](https://github.com/negar-mokhberian/Moral_Foundation_FrameAxis).
