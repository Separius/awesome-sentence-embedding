# awesome-sentence-embedding
A curated list of awesome sentence embedding models

## Table of Contents


* **[General Framework](#general-framework)**
* **[Word Embeddings](#word-embeddings)**
* **[OOV Handling](#oov-handling)**
* **[Contextualized Word Embeddings](#contextualized-word-embeddings)**
* **[Pooling Methods](#pooling-methods)**
* **[Encoders](#encoders)**
* **[Surveys](#surveys)**
* **[Evaluation](#evaluation)**
* **[Multilingual Word Embeddings](#multilingual-word-embeddings)**
* **[Articles](#articles)**

## General Framework

* Almost all the sentence embeddings work like this: given some sort of word embeddings and an optional encoder (for example an LSTM) they obtain the contextualized word embeddings and then they define some sort of pooling (it can be as simple as last pooling) and then based on that they eihter use it directly for the supervised classification task (like infersent) or generate the target sequence (like skip-thought) so in general we have many sentence embeddings that you have never heard of, you can simply do mean-pooling over any word embedding and it's a sentence embedding!

## Word Embeddings

|paper|code|pretrained models|
|---|---|---|
|[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)|[C++](https://github.com/stanfordnlp/GloVe)(official)|[GloVe](https://nlp.stanford.edu/projects/glove/)|
|[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)|[C++](https://code.google.com/archive/p/word2vec/)(official)|[Word2Vec](https://code.google.com/archive/p/word2vec/)|
|[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)|[C++](https://github.com/facebookresearch/fastText)(official)|[fastText](https://fasttext.cc/docs/en/english-vectors.html)|
|[BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages](https://arxiv.org/pdf/1710.02187.pdf)|[Python](https://github.com/bheinzerling/bpemb)(official)|[bpemb](https://github.com/bheinzerling/bpemb#downloads-for-each-language)|
|[ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://arxiv.org/pdf/1612.03975.pdf)|[Python](https://github.com/commonsense/conceptnet-numberbatch)(official)|[Numberbatch](https://github.com/commonsense/conceptnet-numberbatch#downloads)|
