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
|[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)|[C++](https://github.com/stanfordnlp/GloVe)(official)|[models](https://nlp.stanford.edu/projects/glove/)|
