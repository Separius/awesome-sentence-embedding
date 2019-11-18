# awesome-sentence-embedding [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

[![Build Status](https://travis-ci.com/Separius/awesome-sentence-embedding.svg?branch=master)](https://travis-ci.com/Separius/awesome-sentence-embedding)
[![GitHub - LICENSE](https://img.shields.io/github/license/Separius/awesome-sentence-embedding.svg?style=flat)](./LICENSE)

A curated list of pretrained sentence and word embedding models

Update: I won't be able to update the repo for a while, because I [don't have internet access](https://techcrunch.com/2019/11/17/iran-shuts-down-countrys-internet-in-the-wake-of-fuel-protests/)

## Table of Contents

* **[About This Repo](#about-this-repo)**
* **[General Framework](#general-framework)**
* **[Word Embeddings](#word-embeddings)**
* **[OOV Handling](#oov-handling)**
* **[Contextualized Word Embeddings](#contextualized-word-embeddings)**
* **[Pooling Methods](#pooling-methods)**
* **[Encoders](#encoders)**
* **[Evaluation](#evaluation)**
* **[Misc](#misc)**
* **[Vector Mapping](#vector-mapping)**
* **[Articles](#articles)**

## About This Repo

* well there are some awesome-lists for word embeddings and sentence embeddings, but all of them are outdated and more importantly incomplete
* this repo will also be incomplete, but I'll try my best to find and include all the papers with pretrained models
* this is not a typical awesome list because it has tables but I guess it's ok and much better than just a huge list
* if you find any mistakes or find another paper or anything please send a pull request and help me to keep this list up to date
* enjoy!

## General Framework

* Almost all the sentence embeddings work like this:
* Given some sort of word embeddings and an optional encoder (for example an LSTM) they obtain the contextualized word embeddings.
* Then they define some sort of pooling (it can be as simple as last pooling).
* Based on that they either use it directly for the supervised classification task (like infersent) or generate the target sequence (like skip-thought).
* So, in general, we have many sentence embeddings that you have never heard of, you can simply do mean-pooling over any word embedding and it's a sentence embedding!

## Word Embeddings

* Note: don't worry about the language of the code, you can almost always (except for the subword models) just use the pretrained embedding table in the framework of your choice and ignore the training code

{{{word-embedding-table}}}

## OOV Handling

* Drop OOV words!
* One OOV vector(unk vector)
* Use subword models(ngram, bpe, char)
* [ALaCarte](https://github.com/NLPrinceton/ALaCarte): [A La Carte Embedding: Cheap but Effective Induction of Semantic Feature Vectors](http://aclweb.org/anthology/P18-1002)
* [Mimick](https://github.com/yuvalpinter/Mimick): [Mimicking Word Embeddings using Subword RNNs](http://www.aclweb.org/anthology/D17-1010)
* [CompactReconstruction](https://github.com/losyer/compact_reconstruction): [Subword-based Compact Reconstruction of Word Embeddings](https://www.aclweb.org/anthology/N19-1353)

## Contextualized Word Embeddings

* Note: all the unofficial models can load the official pretrained models

{{{contextualized-table}}}

## Pooling Methods

* {Last, Mean, Max}-Pooling
* Special Token Pooling (like BERT and OpenAI's Transformer)
* [SIF](https://github.com/PrincetonML/SIF): [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx)
* [TF-IDF](https://github.com/iarroyof/sentence_embedding): [Unsupervised Sentence Representations as Word Information Series: Revisiting TF--IDF](https://arxiv.org/abs/1710.06524)
* [P-norm](https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings): [Concatenated Power Mean Word Embeddings as Universal Cross-Lingual Sentence Representations](https://arxiv.org/abs/1803.01400)
* [DisC](https://github.com/NLPrinceton/text_embedding): [A Compressed Sensing View of Unsupervised Text Embeddings, Bag-of-n-Grams, and LSTMs](https://openreview.net/pdf?id=B1e5ef-C-)
* [GEM](https://github.com/fursovia/geometric_embedding): [Zero-Training Sentence Embedding via Orthogonal Basis](https://arxiv.org/abs/1810.00438)
* [SWEM](https://github.com/dinghanshen/SWEM): [Baseline Needs More Love: On Simple Word-Embedding-Based Modelsand Associated Pooling Mechanisms](https://arxiv.org/abs/1805.09843)
* [VLAWE](https://github.com/raduionescu/vlawe-boswe/): [Vector of Locally-Aggregated Word Embeddings (VLAWE): A Novel Document-level Representation](https://arxiv.org/abs/1902.08850)
* [Efficient Sentence Embedding using Discrete Cosine Transform](https://arxiv.org/abs/1909.03104)

## Encoders

{{{encoder-table}}}

## Evaluation

* [decaNLP](https://github.com/salesforce/decaNLP): [The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/abs/1806.08730)
* [SentEval](https://github.com/facebookresearch/SentEval): [SentEval: An Evaluation Toolkit for Universal Sentence Representations](https://arxiv.org/abs/1803.05449)
* [GLUE](https://github.com/nyu-mll/GLUE-baselines): [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
* [Exploring Semantic Properties of Sentence Embeddings](http://aclweb.org/anthology/P18-2100)
* [Fine-grained Analysis of Sentence Embeddings Using Auxiliary Prediction Tasks](https://arxiv.org/abs/1608.04207)
* [Word Embeddings Benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks): [How to evaluate word embeddings? On importance of data efficiency and simple supervised tasks](https://arxiv.org/abs/1702.02170)
* [MLDoc](https://github.com/facebookresearch/MLDoc): [A Corpus for Multilingual Document Classification in Eight Languages](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf)
* [LexNET](https://github.com/tensorflow/models/tree/master/research/lexnet_nc): [Olive Oil Is Made of Olives, Baby Oil Is Made for Babies: Interpreting Noun Compounds Using Paraphrases in a Neural Model](https://arxiv.org/abs/1803.08073)
* [wordvectors.net](https://github.com/mfaruqui/eval-word-vectors): [Community Evaluation and Exchange of Word Vectors at wordvectors.org](https://www.manaalfaruqui.com/papers/acl14-vecdemo.pdf)
* [jiant](https://github.com/jsalt18-sentence-repl/jiant): [Looking for ELMo's friends: Sentence-Level Pretraining Beyond Language Modeling](https://arxiv.org/abs/1812.10860)
* [jiant](https://github.com/jsalt18-sentence-repl/jiant): [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/pdf?id=SJzSgnRcKX)
* [Evaluation of sentence embeddings in downstream and linguistic probing tasks](https://arxiv.org/abs/1806.06259)
* [QVEC](https://github.com/ytsvetko/qvec): [Evaluation of Word Vector Representations by Subspace Alignment](http://aclweb.org/anthology/D15-1243)
* [Grammatical Analysis of Pretrained Sentence Encoders with Acceptability Judgments](https://arxiv.org/abs/1901.03438)
* [EQUATE : A Benchmark Evaluation Framework for Quantitative Reasoning in Natural Language Inference](https://arxiv.org/abs/1901.03735)
* [Evaluating Word Embedding Models: Methods andExperimental Results](https://arxiv.org/abs/1901.09785)
* [How to (Properly) Evaluate Cross-Lingual Word Embeddings: On Strong Baselines, Comparative Analyses, and Some Misconceptions](https://arxiv.org/abs/1902.00508)
* [Linguistic Knowledge and Transferability of Contextual Representations](https://homes.cs.washington.edu/~nfliu/papers/liu+gardner+belinkov+peters+smith.naacl2019.pdf): [contextual-repr-analysis](https://github.com/nelson-liu/contextual-repr-analysis)
* [LINSPECTOR](https://github.com/UKPLab/linspector): [Multilingual Probing Tasks for Word Representations](https://arxiv.org/abs/1903.09442)
* [Pitfalls in the Evaluation of Sentence Embeddings](https://arxiv.org/abs/1906.01575)
* [Probing Multilingual Sentence Representations With X-Probe](https://arxiv.org/abs/1906.05061): [xprobe](https://github.com/ltgoslo/xprobe)

## Misc

* [Word Embedding Dimensionality Selection](https://github.com/ziyin-dl/word-embedding-dimensionality-selection): [On the Dimensionality of Word Embedding](https://arxiv.org/abs/1812.04224)
* [Half-Size](https://github.com/vyraun/Half-Size): [Simple and Effective Dimensionality Reduction for Word Embeddings](https://arxiv.org/abs/1708.03629)
* [magnitude](https://github.com/plasticityai/magnitude): [Magnitude: A Fast, Efficient Universal Vector Embedding Utility Package](https://arxiv.org/abs/1810.11190)
* [To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks](https://arxiv.org/abs/1903.05987)
* [Don't Settle for Average, Go for the Max: Fuzzy Sets and Max-Pooled Word Vectors](https://arxiv.org/abs/1904.13264): [fuzzymax](https://github.com/Babylonpartners/fuzzymax)
* [The Pupil Has Become the Master: Teacher-Student Model-BasedWord Embedding Distillation with Ensemble Learning](https://arxiv.org/abs/1906.00095): [EmbeddingDistillation](https://github.com/bgshin/distill_demo)
* [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://www.aclweb.org/anthology/Q15-1016): [hyperwords](https://bitbucket.org/omerlevy/hyperwords/src/default/)
* [Misspelling Oblivious Word Embeddings](https://arxiv.org/abs/1905.09755): [moe](https://github.com/facebookresearch/moe)
* [Single Training Dimension Selection for Word Embedding with
    PCA](https://arxiv.org/abs/1909.01761)
* [Compressing Word Embeddings via Deep Compositional Code Learning](https://arxiv.org/abs/1711.01068): [neuralcompressor](https://github.com/zomux/neuralcompressor)
* [UER: An Open-Source Toolkit for Pre-training
    Models](https://arxiv.org/abs/1909.05658): [UER-py](https://github.com/dbiir/UER-py)
* [Situating Sentence Embedders with Nearest Neighbor
    Overlap](https://arxiv.org/abs/1909.10724)
* [German BERT](https://deepset.ai/german-bert)

## Vector Mapping

* [Cross-lingual Word Vectors Projection Using CCA](https://github.com/mfaruqui/crosslingual-cca): [Improving Vector Space Word Representations Using Multilingual Correlation](http://www.aclweb.org/anthology/E14-1049)
* [vecmap](https://github.com/artetxem/vecmap): [A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings](https://arxiv.org/abs/1805.06297)
* [MUSE](https://github.com/facebookresearch/MUSE): [Unsupervised Machine Translation Using Monolingual Corpora Only](https://arxiv.org/abs/1711.00043)
* [CrossLingualELMo](https://github.com/TalSchuster/CrossLingualELMo): [Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing](https://arxiv.org/abs/1902.09492)

## Articles

* [Comparing Sentence Similarity Methods](http://nlp.town/blog/sentence-similarity/)
* [The Current Best of Universal Word Embeddings and Sentence Embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
* [On sentence representations, pt. 1: what can you fit into a single #$!%@*&% blog post?](https://supernlp.github.io/2018/11/26/sentreps/)
* [Deep-learning-free Text and Sentence Embedding, Part 1](https://www.offconvex.org/2018/06/17/textembeddings/)
* [Deep-learning-free Text and Sentence Embedding, Part 2](https://www.offconvex.org/2018/06/25/textembeddings/)
* [An Overview of Sentence Embedding Methods](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/)
* [Word embeddings in 2017: Trends and future directions](http://ruder.io/word-embeddings-2017/)
* [A Walkthrough of InferSent – Supervised Learning of Sentence Embeddings](https://yashuseth.blog/2018/08/06/infersent-supervised-learning-of-sentence-embeddings/)
* [A survey of cross-lingual word embedding models](http://ruder.io/cross-lingual-embeddings/)
* [Introducing state of the art text classification with universal language models](http://nlp.fast.ai/classification/2018/05/15/introducing-ulmfit.html)
* [Document Embedding Techniques](https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d)
