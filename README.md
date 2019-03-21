# awesome-sentence-embedding [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

[![Build Status](https://travis-ci.com/Separius/awesome-sentence-embedding.svg?branch=master)](https://travis-ci.com/Separius/awesome-sentence-embedding)
[![GitHub - LICENSE](https://img.shields.io/github/license/Separius/awesome-sentence-embedding.svg?style=flat)](./LICENSE)
[![HitCount](http://hits.dwyl.io/Separius/awesome-sentence-embedding.svg)](http://hits.dwyl.io/Separius/awesome-sentence-embedding)

A curated list of pretrained sentence and word embedding models

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
* if you find any mistakes or find another paper or anything please send a pull request
* citation counts were last updated at 2019-03-21 13:40:47.408521
* enjoy!

## General Framework

* Almost all the sentence embeddings work like this:
* Given some sort of word embeddings and an optional encoder (for example an LSTM) they obtain the contextualized word embeddings.
* Then they define some sort of pooling (it can be as simple as last pooling).
* Based on that they either use it directly for the supervised classification task (like infersent) or generate the target sequence (like skip-thought).
* So, in general, we have many sentence embeddings that you have never heard of, you can simply do mean-pooling over any word embedding and it's a sentence embedding!

## Word Embeddings

* Note: don't worry about the language of the code, you can almost always (except for the subword models) just use the pretrained embedding table in the framework of your choice and ignore the training code

|date|paper|citation count|training code|pretrained models|
|:---:|:---:|:---:|:---:|:---:|
|2013/01|[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)|999+|[C](https://github.com/tmikolov/word2vec )|[Word2Vec](https://code.google.com/archive/p/word2vec/ )|
|2014/??|[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)|999+|[C](https://github.com/stanfordnlp/GloVe )|[GloVe](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors )|
|2016/07|[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)|846|[C++](https://github.com/facebookresearch/fastText )|[fastText](https://fasttext.cc/docs/en/english-vectors.html )|
|2017/05|[Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039)|84|[Pytorch](https://github.com/facebookresearch/poincare-embeddings )|-|
|2017/09|[Hash Embeddings for Efficient Word Representations](https://arxiv.org/abs/1709.03933)|4|[Keras](https://github.com/dsv77/hashembedding )|-|
|2017/??|[Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics](http://www.aclweb.org/anthology/D17-1023)|5|[C](https://github.com/zhezhaoa/ngram2vec )|-|
|2018/04|[Representation Tradeoffs for Hyperbolic Embeddings](https://arxiv.org/abs/1804.03329)|12|[Pytorch](https://github.com/HazyResearch/hyperbolics )|[h-MDS](https://github.com/HazyResearch/hyperbolics )|
|2018/04|[Dynamic Meta-Embeddings for Improved Sentence Representations](https://arxiv.org/abs/1804.07983)|3|[Pytorch](https://github.com/facebookresearch/DME )|[DME/CDME](https://github.com/facebookresearch/DME#pre-trained-models )|
|2018/??|[Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings](https://ai.tencent.com/ailab/media/publications/naacl2018/directional_skip-gram.pdf)|0|-|[ChineseEmbedding](https://ai.tencent.com/ailab/nlp/embedding.html )|

## OOV Handling

* Drop OOV words!
* One OOV vector(unk vector)
* Use subword models(ngram, bpe, char)
* [ALaCarte](https://github.com/NLPrinceton/ALaCarte): [A La Carte Embedding: Cheap but Effective Induction of Semantic Feature Vectors](http://aclweb.org/anthology/P18-1002)
* [Mimick](https://github.com/yuvalpinter/Mimick): [Mimicking Word Embeddings using Subword RNNs](http://www.aclweb.org/anthology/D17-1010)

## Contextualized Word Embeddings

* Note: all the unofficial models can load the official pretrained models

|date|paper|citation count|code|pretrained models|
|:---:|:---:|:---:|:---:|:---:|
|-|[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)|N/A|[TF](https://github.com/openai/gpt-2 )<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT )|[GPT-2](https://github.com/openai/gpt-2 )|
|2018/10|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)|82|[TF](https://github.com/google-research/bert )<br>[Keras](https://github.com/Separius/BERT-keras )<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT )<br>[MXNet](https://github.com/imgarylai/bert-embedding )<br>[PaddlePaddle](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE )|BERT([BERT](https://github.com/google-research/bert#pre-trained-models), [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE))|
|2018/??|[Contextual String Embeddings for Sequence Labeling](https://alanakbik.github.io/papers/coling2018.pdf)|4|[Pytorch](https://github.com/zalandoresearch/flair )|[Flair](https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py#L407 )|
|2019/01|[Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)|1|[Pytorch](https://github.com/facebookresearch/XLM )|[XLM](https://github.com/facebookresearch/XLM#pretrained-models )|
|2019/01|[Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/1901.11504)|1|[Pytorch](https://github.com/namisan/mt-dnn )|[MT-DNN](https://github.com/namisan/mt-dnn/blob/master/download.sh )|
|2019/01|[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)|2|[TF](https://github.com/kimiyoung/transformer-xl/tree/master/tf )<br>[Pytorch](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch )<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT )|[Transformer-XL](https://github.com/kimiyoung/transformer-xl/tree/master/tf )|

## Pooling Methods

* {Last, Mean, Max}-Pooling
* Special Token Pooling (like BERT and OpenAI's Transformer)
* [SIF](https://github.com/PrincetonML/SIF): [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx)
* [TF-IDF](https://github.com/iarroyof/sentence_embedding): [Unsupervised Sentence Representations as Word Information Series: Revisiting TF--IDF](https://arxiv.org/abs/1710.06524)
* [P-norm](https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings): [Concatenated Power Mean Word Embeddings as Universal Cross-Lingual Sentence Representations](https://arxiv.org/abs/1803.01400)
* [DisC](https://github.com/NLPrinceton/text_embedding): [A Compressed Sensing View of Unsupervised Text Embeddings, Bag-of-n-Grams, and LSTMs](https://openreview.net/pdf?id=B1e5ef-C-)
* [GEM](https://github.com/fursovia/geometric_embedding): [Zero-Training Sentence Embedding via Orthogonal Basis](https://arxiv.org/abs/1810.00438)
* [SWEM](https://github.com/dinghanshen/SWEM): [Baseline Needs More Love: On Simple Word-Embedding-Based Modelsand Associated Pooling Mechanisms](https://arxiv.org/abs/1805.09843)

## Encoders

|date|paper|citation count|code|model_name|
|:---:|:---:|:---:|:---:|:---:|
|2015/06|[Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](https://arxiv.org/abs/1506.06724)|195|[Theano](https://github.com/ryankiros/skip-thoughts )<br>[TF](https://github.com/tensorflow/models/tree/master/research/skip_thoughts )<br>[Pytorch, Torch](https://github.com/Cadene/skip-thoughts.torch )|SkipThought|
|2017/08|[Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](https://arxiv.org/abs/1708.00524)|N/A|[Keras](https://github.com/bfelbo/DeepMoji )<br>[Pytorch](https://github.com/huggingface/torchMoji )|DeepMoji|
|2017/09|[StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856)|30|[C++](https://github.com/facebookresearch/StarSpace )|StarSpace|
|2017/10|[: Learning Sentence Representations from Explicit Discourse Relations](https://arxiv.org/abs/1710.04334)|22|[Pytorch](https://github.com/windweller/DisExtract )|DisSent|
|2018/03|[Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)|47|[TF-Hub](https://tfhub.dev/google/universal-sentence-encoder/2 )|USE|
|2018/04|[Learning general purpose distributed sentence representations via large scale multi-task learning](https://arxiv.org/abs/1804.00079)|49|[Pytorch](https://github.com/Maluuba/gensen )|GenSen|
|2018/07|[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)|28|[Keras](https://github.com/davidtellez/contrastive-predictive-coding )|CPC|
|2018/10|[Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model](https://arxiv.org/abs/1810.12836)|0|[TF-Hub](https://tfhub.dev/s?q=universal-sentence-encoder-xling )|USE-xling|
|2018/11|[A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks](https://arxiv.org/abs/1811.06031)|2|[Pytorch](https://github.com/huggingface/hmtl )|HMTL|
|2018/12|[Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464)|1|[Pytorch](https://github.com/facebookresearch/LASER )|LASER|

## Evaluation

* [decaNLP](https://github.com/salesforce/decaNLP): [The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/abs/1806.08730)
* [SentEval](https://github.com/facebookresearch/SentEval): [SentEval: An Evaluation Toolkit for Universal Sentence Representations](https://arxiv.org/abs/1803.05449)
* [GLUE](https://github.com/nyu-mll/GLUE-baselines): [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
* [jiant](https://github.com/jsalt18-sentence-repl/jiant): [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/pdf?id=SJzSgnRcKX)
* [How to (Properly) Evaluate Cross-Lingual Word Embeddings: On Strong Baselines, Comparative Analyses, and Some Misconceptions](https://arxiv.org/abs/1902.00508)
* [Exploring Semantic Properties of Sentence Embeddings](http://aclweb.org/anthology/P18-2100)
* [MLDoc](https://github.com/facebookresearch/MLDoc): [A Corpus for Multilingual Document Classification in Eight Languages](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf)
* [Evaluation of sentence embeddings in downstream and linguistic probing tasks](https://arxiv.org/abs/1806.06259)

[//]: # (* [Fine-grained Analysis of Sentence Embeddings Using Auxiliary Prediction Tasks](https://arxiv.org/abs/1608.04207))
[//]: # (* [Word Embeddings Benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks): [How to evaluate word embeddings? On importance of data efficiency and simple supervised tasks](https://arxiv.org/abs/1702.02170))
[//]: # (* [LexNET](https://github.com/tensorflow/models/tree/master/research/lexnet_nc): [Olive Oil Is Made of Olives, Baby Oil Is Made for Babies: Interpreting Noun Compounds Using Paraphrases in a Neural Model](https://arxiv.org/abs/1803.08073))
[//]: # (* [wordvectors.net](https://github.com/mfaruqui/eval-word-vectors): [Community Evaluation and Exchange of Word Vectors at wordvectors.org](https://www.manaalfaruqui.com/papers/acl14-vecdemo.pdf))
[//]: # (* [jiant](https://github.com/jsalt18-sentence-repl/jiant): [Looking for ELMo's friends: Sentence-Level Pretraining Beyond Language Modeling](https://arxiv.org/abs/1812.10860))
[//]: # (* [QVEC](https://github.com/ytsvetko/qvec): [Evaluation of Word Vector Representations by Subspace Alignment](http://aclweb.org/anthology/D15-1243))
[//]: # (* [Grammatical Analysis of Pretrained Sentence Encoders with Acceptability Judgments](https://arxiv.org/abs/1901.03438))
[//]: # (* [EQUATE : A Benchmark Evaluation Framework for Quantitative Reasoning in Natural Language Inference](https://arxiv.org/abs/1901.03735))
[//]: # (* [Evaluating Word Embedding Models: Methods andExperimental Results](https://arxiv.org/abs/1901.09785))

## Misc

* [Word Embedding Dimensionality Selection](https://github.com/ziyin-dl/word-embedding-dimensionality-selection): [On the Dimensionality of Word Embedding](https://arxiv.org/abs/1812.04224)
* [To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks](https://arxiv.org/abs/1903.05987)

[//]: # (* [Half-Size](https://github.com/vyraun/Half-Size): [Simple and Effective Dimensionality Reduction forWord Embeddings](https://arxiv.org/abs/1708.03629))
[//]: # (* [magnitude](https://github.com/plasticityai/magnitude): [Magnitude: A Fast, Efficient Universal Vector Embedding Utility Package](https://arxiv.org/abs/1810.11190))

## Vector Mapping

* [vecmap](https://github.com/artetxem/vecmap): [A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings](https://arxiv.org/abs/1805.06297)
* [MUSE](https://github.com/facebookresearch/MUSE): [Unsupervised Machine Translation Using Monolingual Corpora Only](https://arxiv.org/abs/1711.00043)
* [CrossLingualELMo](https://github.com/TalSchuster/CrossLingualELMo): [Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing](https://arxiv.org/abs/1902.09492)

[//]: # (* [Cross-lingual Word Vectors Projection Using CCA](https://github.com/mfaruqui/crosslingual-cca): [Improving Vector Space Word Representations Using Multilingual Correlation](http://www.aclweb.org/anthology/E14-1049))

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
* [Introducing state of the art text classification with universal language models](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
