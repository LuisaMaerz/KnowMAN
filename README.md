# KnowMAN
### KnowMAN: Weakly Supervised Multinomial Adversarial Networks
[![Python Version](https://img.shields.io/badge/python-3.7-yellow.svg)](https://www.python.org/downloads/release/python-360/)

This repository contains code that is used in our paper: </br>
[KnowMAN: Weakly Supervised Multinomial Networks](https://arxiv.org/abs/2109.07994) - to be published at EMNLP 2021. 🎉  </br>
by Luisa März, Ehsaneddin Asgari, Fabienne Braune, Franziska Zimmermann and Benjamin Roth.


For any questions please [get in touch](mailto:luisa.maerz@volkswagen.de)

---
## What is KnowMAN about?  🤓 

The absence of labeled data for training neural
models is often addressed by leveraging
knowledge about the specific task, resulting
in heuristic but noisy labels. The knowledge
is captured in labeling functions, which detect
certain regularities or patterns in the training
samples and annotate corresponding labels for
training. This process of weakly supervised
training may result in an over-reliance on the
signals captured by the labeling functions and
hinder models to exploit other signals or to
generalize well. 

**KnowMAN** is an
adversarial scheme that enables to control influence
of signals associated with specific labeling
functions. **KnowMAN** forces the network
to learn representations that are invariant
to those signals and to pick up other signals
that are more generally associated with an
output label. **KnowMAN** strongly improves
results compared to direct weakly supervised
learning with a pre-trained transformer language
model and a feature-based baseline.

---

## Usage 🚀 


Experiments described in our paper can be found in the **experiments** folder. 
To run them execute the respective file. </br>
**Please make sure that you have downloaded the data files in advance (see datasets section) and adjusted the datafile path in the yaml files!**


E.g. run the imdb tfidf training:
```
python ./experiments/imdb/train_tfidf_imdb.py
``` 


E.g. run the spam DistilBERT training:

```
python ./experiments/spam/train_transformers_spam.py
```


If you want to change hyperparameters just edit the **yaml** files in the experiments folder. 

---

Baselines can be found in the **baselines** folder. To run them please pass the yaml file for the experiment you want to try here. 

E.g. run the spouse snorkel training:
```
python ./baselines/snorkel_training_knodle.py ./experiments/spouse/spouse_tfidf.yaml
```


Please note that the baselines are only implemented for tf-idf encoding here. The results for DistilBERT baselines can be reproduced by using [Knodle](https://github.com/knodle/knodle).

---
## Datasets 📚 

Datasets used in our work:

- Spam Dataset - a dataset, based on the YouTube comments dataset from [Alberto et al. (2015)](https://www.researchgate.net/publication/300414679_TubeSpam_Comment_Spam_Filtering_on_YouTube). Here, the task is to classify whether a text is relevant to the video or holds spam, such as advertisement.
- Spouse Dataset - relation extraction dataset is based on the Signal Media One-Million News Articles Dataset from [Corney et al. (2016)](http://ceur-ws.org/Vol-1568/paper8.pdf). 
- IMDb Dataset - a dataset, that consists of short movie reviews. The task is to determine whether a review holds a positive or negative sentiment. 

All datasets are part of the the [Knodle](https://github.com/knodle/knodle) framework and can be dowloaded [here](https://knodle.cc/minio/knodle/).




---
## Citation 📑 

When using our work please cite our Acl Anthology print:

```
@inproceedings{marz-etal-2021-knowman,
    title = "{K}now{MAN}: Weakly Supervised Multinomial Adversarial Networks",
    author = {M{\"a}rz, Luisa  and
      Asgari, Ehsaneddin  and
      Braune, Fabienne  and
      Zimmermann, Franziska  and
      Roth, Benjamin},
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.751",
    pages = "9549--9557",
    abstract = "The absence of labeled data for training neural models is often addressed by leveraging knowledge about the specific task, resulting in heuristic but noisy labels. The knowledge is captured in labeling functions, which detect certain regularities or patterns in the training samples and annotate corresponding labels for training. This process of weakly supervised training may result in an over-reliance on the signals captured by the labeling functions and hinder models to exploit other signals or to generalize well. We propose KnowMAN, an adversarial scheme that enables to control influence of signals associated with specific labeling functions. KnowMAN forces the network to learn representations that are invariant to those signals and to pick up other signals that are more generally associated with an output label. KnowMAN strongly improves results compared to direct weakly supervised learning with a pre-trained transformer language model and a feature-based baseline.",
}

```

## Acknowledgments 💎 

This research was funded by the WWTF though the project “Knowledge-infused Deep Learning for Natural Language Processing” (WWTF Vienna Research Group VRG19-008).

