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

|date|paper|citation count|training code|pretrained models|
|:---:|:---:|:---:|:---:|:---:|
|-|[WebVectors: A Toolkit for Building Web Interfaces for Vector Semantic Models](https://rusvectores.org/static/data/webvectors_aist.pdf)|N/A|-|[RusVectōrēs](http://rusvectores.org/en/models/ )|
|2013/01|[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)|999+|[C](https://github.com/tmikolov/word2vec )|[Word2Vec](https://code.google.com/archive/p/word2vec/ )|
|2014/12|[Word Representations via Gaussian Embedding](https://arxiv.org/abs/1412.6623)|111|[Cython](https://github.com/seomoz/word2gauss )|-|
|2014/??|[Dependency-Based Word Embeddings](http://www.aclweb.org/anthology/P14-2050)|453|[C++](https://bitbucket.org/yoavgo/word2vecf/src/default/ )|[word2vecf](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/ )|
|2014/??|[A Probabilistic Model for Learning Multi-Prototype Word Embeddings](http://www.aclweb.org/anthology/C14-1016)|74|[DMTK](https://github.com/Microsoft/distributed_skipgram_mixture )|-|
|2014/??|[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)|999+|[C](https://github.com/stanfordnlp/GloVe )|[GloVe](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors )|
|2015/06|[From Paraphrase Database to Compositional Paraphrase Model and Back](https://arxiv.org/abs/1506.03487)|0|[Theano](https://github.com/jwieting/paragram-word )|[PARAGRAM](http://ttic.uchicago.edu/~wieting/paragram-word-demo.zip )|
|2015/06|[Non-distributional Word Vector Representations](https://arxiv.org/abs/1506.05230)|35|[Python](https://github.com/mfaruqui/non-distributional )|[WordFeat](https://github.com/mfaruqui/non-distributional/blob/master/binary-vectors.txt.gz )|
|2015/06|[Sparse Overcomplete Word Vector Representations](https://arxiv.org/abs/1506.02004)|63|[C++](https://github.com/mfaruqui/sparse-coding )|-|
|2015/??|[Topical Word Embeddings](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9314/9535)|114|[Cython](https://github.com/largelymfs/topical_word_embeddings )|[]( )|
|2015/??|[SensEmbed: Learning Sense Embeddings for Word and Relational Similarity](http://www.aclweb.org/anthology/P/P15/P15-1010.pdf)|116|-|[SensEmbed](http://lcl.uniroma1.it/sensembed/sensembed_vectors.gz )|
|2015/??|[Joint Learning of Character and Word Embeddings](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/ijcai2015_character.pdf)|82|[C](https://github.com/Leonard-Xu/CWE )|-|
|2016/02|[Swivel: Improving Embeddings by Noticing What's Missing](https://arxiv.org/abs/1602.02215)|29|[TF](https://github.com/tensorflow/models/tree/master/research/swivel )|-|
|2016/03|[Counter-fitting Word Vectors to Linguistic Constraints](https://arxiv.org/abs/1603.00892)|89|[Python](https://github.com/nmrksic/counter-fitting )|[counter-fitting](http://mi.eng.cam.ac.uk/~nm480/counter-fitted-vectors.txt.zip )(broken)|
|2016/05|[Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/abs/1605.02019)|12|[Chainer](https://github.com/cemoody/lda2vec )|-|
|2016/06|[Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations](https://arxiv.org/abs/1606.00819)|27|[Go](https://github.com/alexandres/lexvec )|[lexvec](https://github.com/alexandres/lexvec#pre-trained-vectors )|
|2016/06|[Siamese CBOW: Optimizing Word Embeddings for Sentence Representations](https://arxiv.org/abs/1606.04640)|72|[Theano](https://bitbucket.org/TomKenter/siamese-cbow/src/master/ )|[Siamese CBOW](https://bitbucket.org/TomKenter/siamese-cbow/src/master/ )|
|2016/07|[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)|828|[C++](https://github.com/facebookresearch/fastText )|[fastText](https://fasttext.cc/docs/en/english-vectors.html )|
|2016/08|[Morphological Priors for Probabilistic Neural Word Embeddings](https://arxiv.org/abs/1608.01056)|16|[Theano](https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings )|-|
|2016/11|[A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks](https://arxiv.org/abs/1611.01587)|114|[C++](https://github.com/hassyGo/charNgram2vec )|[charNgram2vec](http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz )|
|2016/12|[ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://arxiv.org/abs/1612.03975)|89|[Python](https://github.com/commonsense/conceptnet-numberbatch )|[Numberbatch](https://github.com/commonsense/conceptnet-numberbatch#downloads )|
|2016/??|[Learning Word Meta-Embeddings](http://www.aclweb.org/anthology/P16-1128)|16|-|[Meta-Emb](http://cistern.cis.lmu.de/meta-emb/ )(broken)|
|2017/02|[Offline bilingual word vectors, orthogonal transformations and the inverted softmax](https://arxiv.org/abs/1702.03859)|94|[Python](https://github.com/Babylonpartners/fastText_multilingual )|-|
|2017/04|[Multimodal Word Distributions](https://arxiv.org/abs/1704.08424)|21|[TF](https://github.com/benathi/word2gm )|[word2gm](https://github.com/benathi/word2gm#trained-model )|
|2017/05|[Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039)|82|[Pytorch](https://github.com/facebookresearch/poincare-embeddings )|-|
|2017/06|[Context encoders as a simple but powerful extension of word2vec](https://arxiv.org/abs/1706.02496)|0|[Python](https://github.com/cod3licious/conec )|-|
|2017/06|[Semantic Specialisation of Distributional Word Vector Spaces using Monolingual and Cross-Lingual Constraints](https://arxiv.org/abs/1706.00374)|40|[TF](https://github.com/nmrksic/attract-repel )|[Attract-Repel](https://github.com/nmrksic/attract-repel#available-word-vector-spaces )|
|2017/08|[Making Sense of Word Embeddings](https://arxiv.org/abs/1708.03390)|10|[Python](https://github.com/uhh-lt/sensegram )|[sensegram](http://ltdata1.informatik.uni-hamburg.de/sensegram/ )|
|2017/08|[Learning Chinese Word Representations From Glyphs Of Characters](https://arxiv.org/abs/1708.04755)|6|[C](https://github.com/ray1007/gwe )|-|
|2017/09|[Hash Embeddings for Efficient Word Representations](https://arxiv.org/abs/1709.03933)|4|[Keras](https://github.com/dsv77/hashembedding )|-|
|2017/10|[BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages](https://arxiv.org/abs/1710.02187)|5|[Gensim](https://github.com/bheinzerling/bpemb )|[BPEmb](https://github.com/bheinzerling/bpemb#downloads-for-each-language )|
|2017/11|[SPINE: SParse Interpretable Neural Embeddings](https://arxiv.org/abs/1711.08792)|6|[Pytorch](https://github.com/harsh19/SPINE )|[SPINE](https://drive.google.com/drive/folders/1ksVcWDADmnp0Cl5kezjHqTg3Jnh8q031?usp=sharing )|
|2017/??|[Dict2vec : Learning Word Embeddings using Lexical Dictionaries](http://aclweb.org/anthology/D17-1024)|12|[C++](https://github.com/tca19/dict2vec )|[Dict2vec](https://github.com/tca19/dict2vec#download-pre-trained-vectors )|
|2017/??|[AraVec: A set of Arabic Word Embedding Models for use in Arabic NLP](https://www.researchgate.net/publication/319880027_AraVec_A_set_of_Arabic_Word_Embedding_Models_for_use_in_Arabic_NLP)|12|[Gensim](https://github.com/bakrianoo/aravec )|[AraVec](https://github.com/bakrianoo/aravec#n-grams-models-1 )|
|2017/??|[Joint Embeddings of Chinese Words, Characters, and Fine-grained Subcharacter Components](https://www.cse.ust.hk/~yqsong/papers/2017-EMNLP-ChineseEmbedding.pdf)|14|[C](https://github.com/hkust-knowcomp/jwe )|-|
|2017/??|[Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics](http://www.aclweb.org/anthology/D17-1023)|5|[C](https://github.com/zhezhaoa/ngram2vec )|-|
|2018/04|[Dynamic Meta-Embeddings for Improved Sentence Representations](https://arxiv.org/abs/1804.07983)|3|[Pytorch](https://github.com/facebookresearch/DME )|[DME/CDME](https://github.com/facebookresearch/DME#pre-trained-models )|
|2018/04|[Representation Tradeoffs for Hyperbolic Embeddings](https://arxiv.org/abs/1804.03329)|9|[Pytorch](https://github.com/HazyResearch/hyperbolics )|[h-MDS](https://github.com/HazyResearch/hyperbolics )|
|2018/05|[Analogical Reasoning on Chinese Morphological and Semantic Relations](https://arxiv.org/abs/1805.06504)|4|-|[ChineseWordVectors](https://github.com/Embedding/Chinese-Word-Vectors )|
|2018/06|[Probabilistic FastText for Multi-Sense Word Embeddings](https://arxiv.org/abs/1806.02901)|3|[C++](https://github.com/benathi/multisense-prob-fasttext )|[Probabilistic FastText](https://github.com/benathi/multisense-prob-fasttext#3-loading-and-analyzing-pre-trained-models )|
|2018/09|[FRAGE: Frequency-Agnostic Word Representation](https://arxiv.org/abs/1809.06858)|0|[Pytorch](https://github.com/ChengyueGongR/Frequency-Agnostic )|-|
|2018/12|[Wikipedia2Vec: An Optimized Tool for LearningEmbeddings of Words and Entities from Wikipedia](https://arxiv.org/abs/1812.06280)|0|[Cython](https://github.com/wikipedia2vec/wikipedia2vec )|[Wikipedia2Vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/ )|
|2018/??|[Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings](https://ai.tencent.com/ailab/media/publications/naacl2018/directional_skip-gram.pdf)|0|-|[ChineseEmbedding](https://ai.tencent.com/ailab/nlp/embedding.html )|
|2018/??|[cw2vec: Learning Chinese Word Embeddings with Stroke n-gram Information](http://www.statnlp.org/wp-content/uploads/papers/2018/cw2vec/cw2vec.pdf)|7|[C++](https://github.com/bamtercelboo/cw2vec )|-|

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
|-|[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)|N/A|[TF](https://github.com/openai/gpt-2 )<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT )(unofficial)|[GPT-2](https://github.com/openai/gpt-2 )|
|2017/08|[Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)|122|[Pytorch](https://github.com/salesforce/cove )<br>[Keras](https://github.com/rgsachin/CoVe )(unofficial)|[CoVe](https://github.com/salesforce/cove )|
|2018/01|[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)|54|[Pytorch](https://github.com/fastai/fastai/tree/ulmfit_v1 )(unofficial)|ULMFit([English](link), [Zoo](link))|
|2018/02|[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)|308|[Pytorch](https://github.com/allenai/allennlp )<br>[TF](https://github.com/allenai/bilm-tf )|ELMO([AllenNLP](link), [TF-Hub](link))|
|2018/04|[Efficient Contextualized Representation:Language Model Pruning for Sequence Labeling](https://arxiv.org/abs/1804.07827)|1|[Pytorch](https://github.com/LiyuanLucasLiu/LD-Net )|[LD-Net](https://github.com/LiyuanLucasLiu/LD-Net#language-models )|
|2018/07|[Towards Better UD Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation](https://arxiv.org/abs/1807.03121)|4|[Pytorch](https://github.com/HIT-SCIR/ELMoForManyLangs )|[ELMo](https://github.com/HIT-SCIR/ELMoForManyLangs#downloads )|
|2018/08|[Direct Output Connection for a High-Rank Language Model](https://arxiv.org/abs/1808.10143)|0|[Pytorch](https://github.com/nttcslab-nlp/doc_lm )|[DOC](https://drive.google.com/open?id=1ug-6ISrXHEGcWTk5KIw8Ojdjuww-i-Ci )|
|2018/10|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)|73|[TF](https://github.com/google-research/bert )<br>[Keras](https://github.com/Separius/BERT-keras )(unofficial)<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT )(unofficial)<br>[MXNet](https://github.com/imgarylai/bert-embedding )(unofficial)|[BERT](https://github.com/google-research/bert#pre-trained-models )|
|2018/??|[Contextual String Embeddings for Sequence Labeling]()|2|[Pytorch](https://github.com/zalandoresearch/flair )|[Flair](https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py#L407 )|
|2018/??|[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)|63|[TF](https://github.com/openai/finetune-transformer-lm )<br>[Keras](https://github.com/Separius/BERT-keras )(unofficial)<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT )(unofficial)|[GPT](https://github.com/openai/finetune-transformer-lm )|
|2019/01|[BioBERT: pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746)|0|[TF](https://github.com/dmis-lab/biobert )|[BioBERT](https://github.com/naver/biobert-pretrained )|
|2019/01|[Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)|1|[Pytorch](https://github.com/facebookresearch/XLM )|[XLM](https://github.com/facebookresearch/XLM#pretrained-models )|
|2019/01|[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)|2|[TF](https://github.com/kimiyoung/transformer-xl/tree/master/tf )<br>[Pytorch](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch )<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT )(unofficial)|[Transformer-XL](https://github.com/kimiyoung/transformer-xl/tree/master/tf )|

## Pooling Methods

* {Last, Mean, Max}-Pooling
* Special Token Pooling (like BERT and OpenAI's Transformer)
* [SIF](https://github.com/PrincetonML/SIF): [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx)
* [TF-IDF](https://github.com/iarroyof/sentence_embedding): [Unsupervised Sentence Representations as Word Information Series: Revisiting TF--IDF](https://arxiv.org/pdf/1710.06524)
* [P-norm](https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings): [Concatenated Power Mean Word Embeddings as Universal Cross-Lingual Sentence Representations](https://arxiv.org/pdf/1803.01400)
* [DisC](https://github.com/NLPrinceton/text_embedding): [A Compressed Sensing View of Unsupervised Text Embeddings, Bag-of-n-Grams, and LSTMs](https://openreview.net/pdf?id=B1e5ef-C-)
* [GEM](https://github.com/fursovia/geometric_embedding): [Zero-Training Sentence Embedding via Orthogonal Basis](https://arxiv.org/pdf/1810.00438.pdf)
* [SWEM](https://github.com/dinghanshen/SWEM): [Baseline Needs More Love: On Simple Word-Embedding-Based Modelsand Associated Pooling Mechanisms](https://arxiv.org/pdf/1805.09843.pdf)

## Encoders

{{{encoder-table}}}

## Evaluation

* [decaNLP](https://github.com/salesforce/decaNLP): [The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/pdf/1806.08730)
* [SentEval](https://github.com/facebookresearch/SentEval): [SentEval: An Evaluation Toolkit for Universal Sentence Representations](https://arxiv.org/pdf/1803.05449)
* [GLUE](https://github.com/nyu-mll/GLUE-baselines): [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/pdf/1804.07461.pdf)
* [Exploring Semantic Properties of Sentence Embeddings](http://aclweb.org/anthology/P18-2100)
* [Fine-grained Analysis of Sentence Embeddings Using Auxiliary Prediction Tasks](https://arxiv.org/pdf/1608.04207)
* [Word Embeddings Benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks): [How to evaluate word embeddings? On importance of data efficiency and simple supervised tasks](https://arxiv.org/pdf/1702.02170)
* [MLDoc](https://github.com/facebookresearch/MLDoc): [A Corpus for Multilingual Document Classification in Eight Languages](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf)
* [LexNET](https://github.com/tensorflow/models/tree/master/research/lexnet_nc): [Olive Oil Is Made of Olives, Baby Oil Is Made for Babies: Interpreting Noun Compounds Using Paraphrases in a Neural Model](https://arxiv.org/pdf/1803.08073.pdf)
* [wordvectors.net](https://github.com/mfaruqui/eval-word-vectors): [Community Evaluation and Exchange of Word Vectors at wordvectors.org](https://www.manaalfaruqui.com/papers/acl14-vecdemo.pdf)
* [jiant](https://github.com/jsalt18-sentence-repl/jiant): [Looking for ELMo's friends: Sentence-Level Pretraining Beyond Language Modeling](https://arxiv.org/pdf/1812.10860.pdf)
* [jiant](https://github.com/jsalt18-sentence-repl/jiant): [What do you learn from context? Probing for sentence structure in contextualized word representations](https://openreview.net/pdf?id=SJzSgnRcKX)
* [Evaluation of sentence embeddings in downstream and linguistic probing tasks](https://arxiv.org/pdf/1806.06259)
* [QVEC](https://github.com/ytsvetko/qvec): [Evaluation of Word Vector Representations by Subspace Alignment](http://aclweb.org/anthology/D15-1243)
* [Grammatical Analysis of Pretrained Sentence Encoders with Acceptability Judgments](https://arxiv.org/pdf/1901.03438.pdf)
* [EQUATE : A Benchmark Evaluation Framework for Quantitative Reasoning in Natural Language Inference](https://arxiv.org/pdf/1901.03735.pdf)
* [Evaluating Word Embedding Models: Methods andExperimental Results](https://arxiv.org/pdf/1901.09785.pdf)
* [How to (Properly) Evaluate Cross-Lingual Word Embeddings: On Strong Baselines, Comparative Analyses, and Some Misconceptions](https://arxiv.org/pdf/1902.00508.pdf)

## Misc

* [Word Embedding Dimensionality Selection](https://github.com/ziyin-dl/word-embedding-dimensionality-selection): [On the Dimensionality of Word Embedding](https://arxiv.org/pdf/1812.04224.pdf)
* [Half-Size](https://github.com/vyraun/Half-Size): [Simple and Effective Dimensionality Reduction forWord Embeddings](https://arxiv.org/pdf/1708.03629.pdf)
* [magnitude](https://github.com/plasticityai/magnitude): [Magnitude: A Fast, Efficient Universal Vector Embedding Utility Package](https://arxiv.org/pdf/1810.11190.pdf)

## Vector Mapping

* [Cross-lingual Word Vectors Projection Using CCA](https://github.com/mfaruqui/crosslingual-cca): [Improving Vector Space Word Representations Using Multilingual Correlation](http://www.aclweb.org/anthology/E14-1049)
* [vecmap](https://github.com/artetxem/vecmap): [A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings](https://arxiv.org/pdf/1805.06297)
* [MUSE](https://github.com/facebookresearch/MUSE): [Unsupervised Machine Translation Using Monolingual Corpora Only](https://arxiv.org/pdf/1711.00043)
* [CrossLingualELMo](https://github.com/TalSchuster/CrossLingualELMo): [Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing](https://arxiv.org/pdf/1902.09492.pdf)

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
