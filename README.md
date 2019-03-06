# awesome-sentence-embedding [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

[![Build Status](https://travis-ci.com/Separius/awesome-sentence-embedding.svg?branch=master)](https://travis-ci.com/Separius/awesome-sentence-embedding)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Separius/awesome-sentence-embedding/issues)
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

{{{word-embedding-table}}}

## OOV Handling

* Drop OOV words!
* One OOV vector(unk vector)
* Use subword models(ngram, bpe, char)
* [ALaCarte](https://github.com/NLPrinceton/ALaCarte): [A La Carte Embedding: Cheap but Effective Induction of Semantic Feature Vectors](http://aclweb.org/anthology/P18-1002)
* [Mimick](https://github.com/yuvalpinter/Mimick): [Mimicking Word Embeddings using Subword RNNs](http://www.aclweb.org/anthology/D17-1010)

## Contextualized Word Embeddings

* Note: all the unofficial models can load the official pretrained models

|paper|code|pretrained models|
|---|---|---|
|[Learned in Translation: Contextualized Word Vectors](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf)|[Pytorch](https://github.com/salesforce/cove )(official)<br>[Keras](https://github.com/rgsachin/CoVe)|[CoVe](https://github.com/salesforce/cove)|
|[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)|[Pytorch](https://github.com/fastai/fastai/tree/ulmfit_v1 )(official)|ULMFit([English](https://docs.fast.ai/text.html#Fine-tuning-a-language-model), [Zoo](https://forums.fast.ai/t/language-model-zoo-gorilla/14623/1))|
|[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365)|[Pytorch](https://github.com/allenai/allennlp )(official)<br>[TF](https://github.com/allenai/bilm-tf )(official)|ELMO([AllenNLP](https://allennlp.org/elmo), [TF-Hub](https://tfhub.dev/google/elmo/2))|
|[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)|[TF](https://github.com/openai/finetune-transformer-lm )(official)<br>[Keras](https://github.com/Separius/BERT-keras)<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT)|[GPT](https://github.com/openai/finetune-transformer-lm)
|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)|[TF](https://github.com/google-research/bert )(official)<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT)<br>[Keras](https://github.com/Separius/BERT-keras)<br>[MXNet](https://github.com/imgarylai/bert-embedding)|[BERT](https://github.com/google-research/bert#pre-trained-models)|
|[Towards Better UD Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation](https://arxiv.org/pdf/1807.03121)|[Pytorch](https://github.com/HIT-SCIR/ELMoForManyLangs )(official)|[ELMo](https://github.com/HIT-SCIR/ELMoForManyLangs#downloads)|
|[Contextual String Embeddings for Sequence Labeling](http://aclweb.org/anthology/C18-1139)|[Pytorch](https://github.com/zalandoresearch/flair )(official)|[Flair](https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py#L407)|
|[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860)|[TF](https://github.com/kimiyoung/transformer-xl/tree/master/tf )(official)<br>[Pytorch](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch )(official)<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT)|[Transformer-XL](https://github.com/kimiyoung/transformer-xl/tree/master/tf)|
|[BioBERT: pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/pdf/1901.08746.pdf)|[TF](https://github.com/dmis-lab/biobert )(official)|[BioBERT](https://github.com/naver/biobert-pretrained)|
|[Cross-lingual Language Model Pretraining](https://arxiv.org/pdf/1901.07291)|[Pytorch](https://github.com/facebookresearch/XLM )(official)|[XLM](https://github.com/facebookresearch/XLM#pretrained-models)|
|[Direct Output Connection for a High-Rank Language Model](https://arxiv.org/pdf/1808.10143.pdf)|[Pytorch](https://github.com/nttcslab-nlp/doc_lm )(official)|[DOC](https://drive.google.com/open?id=1ug-6ISrXHEGcWTk5KIw8Ojdjuww-i-Ci)|
|[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)|[Tensorflow](https://github.com/openai/gpt-2 )(official, no training)<br>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT)|[GPT-2-117M](https://github.com/openai/gpt-2)|
|[Efficient Contextualized Representation:Language Model Pruning for Sequence Labeling](https://arxiv.org/pdf/1804.07827.pdf)|[Pytorch](https://github.com/LiyuanLucasLiu/LD-Net )(official)|[LD-Net](https://github.com/LiyuanLucasLiu/LD-Net#language-models)|

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

|paper|code|name|
|---|---|---|
|[An efficient framework for learning sentence representations](https://arxiv.org/pdf/1803.02893.pdf)|[TF](https://github.com/lajanugen/S2V )(official, pretrained)|Quick-Thought|
|[Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](https://arxiv.org/pdf/1708.00524)|[Keras](https://github.com/bfelbo/DeepMoji )(official, pretrained)<br>[Pytorch](https://github.com/huggingface/torchMoji )(load_pretrained)|DeepMoji|
|[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/pdf/1705.02364)|[Pytorch](https://github.com/facebookresearch/InferSent )(official, pretrained)|InferSent|
|[Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/pdf/1812.10464.pdf)|[Pytorch](https://github.com/facebookresearch/LASER )(official, pretrained)|LASER|
|[Learning general purpose distributed sentence representations via large scale multi-task learning](https://arxiv.org/pdf/1804.00079)|[Pytorch](https://github.com/Maluuba/gensen )(official, pretrained)|GenSen|
|[Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053)|[Pytorch](https://github.com/inejc/paragraph-vectors)<br>[Python](https://github.com/jhlau/doc2vec )(pretrained)|Doc2Vec|
|[Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features](https://arxiv.org/pdf/1703.02507.pdf)|[C++](https://github.com/epfml/sent2vec )(official, pretrained)|Sent2Vec|
|[Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](https://arxiv.org/pdf/1506.06724)|[Theano](https://github.com/ryankiros/skip-thoughts )(official, pretrained)<br>[TF](https://github.com/tensorflow/models/tree/master/research/skip_thoughts )(pretrained)<br>[Pytorch,Torch](https://github.com/Cadene/skip-thoughts.torch )(load_pretrained)|SkipThought|
|[Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/pdf/1704.01444)|[TF](https://github.com/openai/generating-reviews-discovering-sentiment )(official, pretrained)<br>[Pytorch](https://github.com/guillitte/pytorch-sentiment-neuron )(load_pretrained)<br>[Pytorch](https://github.com/NVIDIA/sentiment-discovery )(pretrained)|SentimentNeuron|
|[From Word Embeddings to Document Distances](http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf)|[C,Python](https://github.com/mkusner/wmd )(official)|Word Mover's Distance|
|[Word Mover's Embedding: From Word2Vec to Document Embedding](https://arxiv.org/pdf/1811.01713)|[C,Python](https://github.com/IBM/WordMoversEmbeddings )(official)|WordMoversEmbeddings|
|[Convolutional Neural Network for Universal Sentence Embeddings](https://pdfs.semanticscholar.org/d827/32de6336dd6443ff33cccbb92ced0196ecc1.pdf)|[Theano](https://github.com/XiaoqiJiao/COLING2018 )(official, pretrained)|CSE|
|[Towards Universal Paraphrastic Sentence Embeddings](https://arxiv.org/pdf/1511.08198)|[Theano](https://github.com/jwieting/iclr2016 )(official, pretrained)|ParagramPhrase|
|[Charagram: Embedding Words and Sentences via Character n-grams](https://aclweb.org/anthology/D16-1157)|[Theano](https://github.com/jwieting/charagram )(official, pretrained)|Charagram|
|[Revisiting Recurrent Networks for Paraphrastic Sentence Embeddings](https://arxiv.org/pdf/1705.00364)|[Theano](https://github.com/jwieting/acl2017 )(official, pretrained)|GRAN|
|[Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations](https://arxiv.org/pdf/1711.05732)|[Theano](https://github.com/jwieting/para-nmt-50m )(official, pretrained)|para-nmt|
|[Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](https://arxiv.org/pdf/1411.2539)|[Theano](https://github.com/ryankiros/visual-semantic-embedding )(official, pretrained)<br>[Pytorch](https://github.com/linxd5/VSE_Pytorch )(load_pretrained)|VSE|
|[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/pdf/1707.05612)|[Pytorch](https://github.com/fartashf/vsepp )(official, pretrained)|VSE++|
|[End-Task Oriented Textual Entailment via Deep Explorations of Inter-Sentence Interactions](https://arxiv.org/pdf/1804.08813)|[Theano](https://github.com/yinwenpeng/SciTail )(official, pretrained)|DEISTE|
|[Learning Universal Sentence Representations with Mean-Max Attention Autoencoder](http://aclweb.org/anthology/D18-1481)|[TF](https://github.com/Zminghua/SentEncoding )(official, pretrained)|Mean-MaxAAE|
|[BioSentVec: creating sentence embeddings for biomedical texts](https://arxiv.org/pdf/1810.09302)|[Python](https://github.com/ncbi-nlp/BioSentVec )(official, pretrained)|BioSentVec|
|[DisSent: Learning Sentence Representations from Explicit Discourse Relations](https://arxiv.org/pdf/1710.04334.pdf)|[Pytorch](https://github.com/windweller/DisExtract )(official, email_for_pretrained)|DisSent|
|[Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf)|[TF-Hub](https://tfhub.dev/google/universal-sentence-encoder/2 )(official, pretrained)|USE|
|[Learning Distributed Representations of Sentences from Unlabelled Data](https://arxiv.org/pdf/1602.03483)|[Python](https://github.com/fh295/SentenceRepresentation )(official)|FastSent|
|[Embedding Text in Hyperbolic Spaces](https://arxiv.org/pdf/1806.04313.pdf)|[TF](https://github.com/brain-research/hyperbolictext )(official)|HyperText|
|[StarSpace: Embed All The Things!](https://arxiv.org/pdf/1709.03856.pdf)|[C++](https://github.com/facebookresearch/StarSpace )(official)|StarSpace|
|[A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks](https://arxiv.org/pdf/1811.06031.pdf)|[Pytorch](https://github.com/huggingface/hmtl )(official, pretrained)|HMTL|
|[Learning Generic Sentence Representations Using Convolutional Neural Networks](https://arxiv.org/pdf/1611.07897.pdf)|[Theano](https://github.com/zhegan27/ConvSent )(official)|ConvSent|
|[Context Mover’s Distance & Barycenters: Optimal transport of contexts for building representations](https://arxiv.org/pdf/1808.09663.pdf)|[Python](https://github.com/context-mover/context-mover-distance-and-barycenters )(official, pretrained)|CMD|
|[No Training Required: Exploring Random Encoders for Sentence Classification](https://arxiv.org/pdf/1901.10444.pdf)|[Pytorch](https://github.com/facebookresearch/randsent )(official)|randsent|
|[CBOW Is Not All You Need: Combining CBOW with the Compositional Matrix Space Model](https://openreview.net/pdf?id=H1MgjoR9tQ)|[Pytorch](https://github.com/florianmai/word2mat )(official)|CMOW|
|[Order-Embeddings of Images and Language](https://arxiv.org/pdf/1511.06361.pdf)|[Theano](https://github.com/ivendrov/order-embedding )(official, pretrained)|order-embedding|
|[Dual-Path Convolutional Image-Text Embedding with Instance Loss](https://arxiv.org/pdf/1711.05535.pdf)|[Matlab](https://github.com/layumi/Image-Text-Embedding )(official, pretrained)|Image-Text-Embedding|
|[Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model](https://arxiv.org/pdf/1810.12836.pdf)|[TF-Hub](https://tfhub.dev/s?q=universal-sentence-encoder-xling )(official, pretrained)|USE-xling|

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
