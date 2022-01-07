## eMFDscore: Extended Moral Foundation Dictionary Scoring for Python 
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open Source Love png2](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

**eMFDscore** is a library for the fast and flexible extraction of various moral information metrics from textual input data. eMFDscore is built on [spaCy](https://github.com/explosion/spaCy) for faster execution and performs minimal preprocessing consisting of tokenization, syntactic dependency parsing, lower-casing, and stopword/punctuation/whitespace removal. eMFDscore lets users score documents with **multiple Moral Foundations Dictionaries**, provides **various metrics for analyzing moral information**, and extracts **moral patient, agent, and attribute words** related to entities.
    
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

### eMFDscore in Google Colaboratory

eMFDscore can also be run in [google colab](https://colab.research.google.com/notebooks/intro.ipynb). All you need to do is add these lines to the beginning of your notebook, execute them, and then restart your runtime:

```
!pip install -U pip setuptools wheel
!pip install -U spacy
!python -m spacy download en_core_web_sm
!pip install git+https://github.com/medianeuroscience/emfdscore.git
```

You can then use eMFDscore as a python library as documented in our [tutorial](https://github.com/medianeuroscience/emfdscore/blob/master/eMFDscore_Tutorial.ipynb). 

## Usage 
Please refer to this [tutorial](https://github.com/medianeuroscience/emfdscore/blob/master/eMFDscore_Tutorial.ipynb) to learn how to use eMFDscore. 

If you are using the eMFD within the [Global Database of Events, Language, and Tone (GDELT)](https://blog.gdeltproject.org/examining-trends-in-moral-news-framing-across-a-decade-of-television-coverage/) please read the following [documentation](https://github.com/medianeuroscience/emfdscore/blob/master/emfd_gdelt_readme.pdf).  

For using the eMFD on shorter texts (e.g., tweets and news headlines), we suggest to apply the eMFD in a [FrameAxis](https://github.com/negar-mokhberian/Moral_Foundation_FrameAxis).

## Applications 
The eMFD has been used in the following applications (ordered by date of publication):
- [Harris, C., Myers, A., & Kaiser, A. (2022). Being Seen: How Markets Impact Our Moral Sentiments. Available at SSRN: https://ssrn.com/abstract=3997378 or http://dx.doi.org/10.2139/ssrn.3997378](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3997378) 
- [Malik, M., Hopp, F. R., Chen, Y., & Weber, R. (2021). Does Regional Variation in Pathogen Prevalence Predict the Moralization of Language in COVID-19 News? Journal of Language and Social Psychology.](https://doi.org/10.1177%2F0261927X211044194)
- [Chen, Kaiping, Zening Duan, and Sijia Yang. "Twitter as research data: Tools, costs, skill sets, and lessons learned." Politics and the Life Sciences (2021): 1-17.](https://www.cambridge.org/core/journals/politics-and-the-life-sciences/article/twitter-as-research-data/6B31D18C5E2F9B8F9C0301BFB05F1C27)
- [Van Vliet, L. (2021). Moral expressions in 280 characters or less: An Analysis of Politician tweets following the 2016 Brexit referendum vote. Frontiers in Big Data, 4, 49.](https://www.frontiersin.org/articles/10.3389/fdata.2021.699653/full)
- [Priniski, J. H., Mokhberian, N., Harandizadeh, B., Morstatter, F., Lerman, K., Lu, H., & Brantingham, P. J. (2021). Mapping Moral Valence of Tweets Following the Killing of George Floyd. arXiv preprint arXiv:2104.09578.](https://arxiv.org/abs/2104.09578)
- [Hopp, F. R., Fisher, J. T., & Weber, R. (2020). A graph-learning approach for detecting moral conflict in movie scripts. Media and Communication, 8(3), 164.](https://www.cogitatiopress.com/mediaandcommunication/article/view/3155)

