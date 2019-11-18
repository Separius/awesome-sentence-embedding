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

|date|paper|citation count|training code|pretrained models|
|:---:|:---:|:---:|:---:|:---:|
|-|[WebVectors: A Toolkit for Building Web Interfaces for Vector Semantic Models](https://rusvectores.org/static/data/webvectors_aist.pdf)|N/A|-|[RusVectōrēs](http://rusvectores.org/en/models/ )|
|-|[Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings](https://ai.tencent.com/ailab/media/publications/naacl2018/directional_skip-gram.pdf)|N/A|-|[ChineseEmbedding](https://ai.tencent.com/ailab/nlp/embedding.html )|
|2013/01|[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)|999+|[C](https://github.com/tmikolov/word2vec )|[Word2Vec](https://code.google.com/archive/p/word2vec/ )|
|2014/12|[Word Representations via Gaussian Embedding](https://arxiv.org/abs/1412.6623)|131|[Cython](https://github.com/seomoz/word2gauss )|-|
|2014/??|[Dependency-Based Word Embeddings](http://www.aclweb.org/anthology/P14-2050)|579|[C++](https://bitbucket.org/yoavgo/word2vecf/src/default/ )|[word2vecf](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/ )|
|2014/??|[A Probabilistic Model for Learning Multi-Prototype Word Embeddings](http://www.aclweb.org/anthology/C14-1016)|89|[DMTK](https://github.com/Microsoft/distributed_skipgram_mixture )|-|
|2014/??|[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)|999+|[C](https://github.com/stanfordnlp/GloVe )|[GloVe](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors )|
|2015/06|[From Paraphrase Database to Compositional Paraphrase Model and Back](https://arxiv.org/abs/1506.03487)|0|[Theano](https://github.com/jwieting/paragram-word )|[PARAGRAM](http://ttic.uchicago.edu/~wieting/paragram-word-demo.zip )|
|2015/06|[Non-distributional Word Vector Representations](https://arxiv.org/abs/1506.05230)|42|[Python](https://github.com/mfaruqui/non-distributional )|[WordFeat](https://github.com/mfaruqui/non-distributional/blob/master/binary-vectors.txt.gz )|
|2015/06|[Sparse Overcomplete Word Vector Representations](https://arxiv.org/abs/1506.02004)|81|[C++](https://github.com/mfaruqui/sparse-coding )|-|
|2015/??|[Joint Learning of Character and Word Embeddings](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/download/11000/10834)|111|[C](https://github.com/Leonard-Xu/CWE )|-|
|2015/??|[Topical Word Embeddings](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9314/9535)|157|[Cython](https://github.com/largelymfs/topical_word_embeddings )|[]( )|
|2015/??|[SensEmbed: Learning Sense Embeddings for Word and Relational Similarity](http://www.aclweb.org/anthology/P/P15/P15-1010.pdf)|162|-|[SensEmbed](http://lcl.uniroma1.it/sensembed/sensembed_vectors.gz )|
|2016/02|[Swivel: Improving Embeddings by Noticing What's Missing](https://arxiv.org/abs/1602.02215)|47|[TF](https://github.com/tensorflow/models/tree/master/research/swivel )|-|
|2016/03|[Counter-fitting Word Vectors to Linguistic Constraints](https://arxiv.org/abs/1603.00892)|130|[Python](https://github.com/nmrksic/counter-fitting )|[counter-fitting](http://mi.eng.cam.ac.uk/~nm480/counter-fitted-vectors.txt.zip )(broken)|
|2016/05|[Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/abs/1605.02019)|38|[Chainer](https://github.com/cemoody/lda2vec )|-|
|2016/06|[Siamese CBOW: Optimizing Word Embeddings for Sentence Representations](https://arxiv.org/abs/1606.04640)|111|[Theano](https://bitbucket.org/TomKenter/siamese-cbow/src/master/ )|[Siamese CBOW](https://bitbucket.org/TomKenter/siamese-cbow/src/master/ )|
|2016/06|[Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations](https://arxiv.org/abs/1606.00819)|34|[Go](https://github.com/alexandres/lexvec )|[lexvec](https://github.com/alexandres/lexvec#pre-trained-vectors )|
|2016/07|[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)|0|[C++](https://github.com/facebookresearch/fastText )|[fastText](https://fasttext.cc/docs/en/english-vectors.html )|
|2016/08|[Morphological Priors for Probabilistic Neural Word Embeddings](https://arxiv.org/abs/1608.01056)|23|[Theano](https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings )|-|
|2016/11|[A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks](https://arxiv.org/abs/1611.01587)|198|[C++](https://github.com/hassyGo/charNgram2vec )|[charNgram2vec](http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz )|
|2016/12|[ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://arxiv.org/abs/1612.03975)|188|[Python](https://github.com/commonsense/conceptnet-numberbatch )|[Numberbatch](https://github.com/commonsense/conceptnet-numberbatch#downloads )|
|2016/??|[Learning Word Meta-Embeddings](http://www.aclweb.org/anthology/P16-1128)|22|-|[Meta-Emb](http://cistern.cis.lmu.de/meta-emb/ )(broken)|
|2017/02|[Offline bilingual word vectors, orthogonal transformations and the inverted softmax](https://arxiv.org/abs/1702.03859)|171|[Python](https://github.com/Babylonpartners/fastText_multilingual )|-|
|2017/04|[Multimodal Word Distributions](https://arxiv.org/abs/1704.08424)|30|[TF](https://github.com/benathi/word2gm )|[word2gm](https://github.com/benathi/word2gm#trained-model )|
|2017/05|[Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039)|155|[Pytorch](https://github.com/facebookresearch/poincare-embeddings )|-|
|2017/06|[Context encoders as a simple but powerful extension of word2vec](https://arxiv.org/abs/1706.02496)|4|[Python](https://github.com/cod3licious/conec )|-|
|2017/06|[Semantic Specialisation of Distributional Word Vector Spaces using Monolingual and Cross-Lingual Constraints](https://arxiv.org/abs/1706.00374)|69|[TF](https://github.com/nmrksic/attract-repel )|[Attract-Repel](https://github.com/nmrksic/attract-repel#available-word-vector-spaces )|
|2017/08|[Learning Chinese Word Representations From Glyphs Of Characters](https://arxiv.org/abs/1708.04755)|18|[C](https://github.com/ray1007/gwe )|-|
|2017/08|[Making Sense of Word Embeddings](https://arxiv.org/abs/1708.03390)|24|[Python](https://github.com/uhh-lt/sensegram )|[sensegram](http://ltdata1.informatik.uni-hamburg.de/sensegram/ )|
|2017/09|[Hash Embeddings for Efficient Word Representations](https://arxiv.org/abs/1709.03933)|7|[Keras](https://github.com/dsv77/hashembedding )|-|
|2017/10|[BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages](https://arxiv.org/abs/1710.02187)|23|[Gensim](https://github.com/bheinzerling/bpemb )|[BPEmb](https://github.com/bheinzerling/bpemb#downloads-for-each-language )|
|2017/11|[SPINE: SParse Interpretable Neural Embeddings](https://arxiv.org/abs/1711.08792)|19|[Pytorch](https://github.com/harsh19/SPINE )|[SPINE](https://drive.google.com/drive/folders/1ksVcWDADmnp0Cl5kezjHqTg3Jnh8q031?usp=sharing )|
|2017/??|[Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics](http://www.aclweb.org/anthology/D17-1023)|14|[C](https://github.com/zhezhaoa/ngram2vec )|-|
|2017/??|[Dict2vec : Learning Word Embeddings using Lexical Dictionaries](http://aclweb.org/anthology/D17-1024)|25|[C++](https://github.com/tca19/dict2vec )|[Dict2vec](https://github.com/tca19/dict2vec#download-pre-trained-vectors )|
|2017/??|[Joint Embeddings of Chinese Words, Characters, and Fine-grained Subcharacter Components](https://www.cse.ust.hk/~yqsong/papers/2017-EMNLP-ChineseEmbedding.pdf)|26|[C](https://github.com/hkust-knowcomp/jwe )|-|
|2017/??|[AraVec: A set of Arabic Word Embedding Models for use in Arabic NLP](https://www.researchgate.net/publication/319880027_AraVec_A_set_of_Arabic_Word_Embedding_Models_for_use_in_Arabic_NLP)|42|[Gensim](https://github.com/bakrianoo/aravec )|[AraVec](https://github.com/bakrianoo/aravec#n-grams-models-1 )|
|2018/04|[Dynamic Meta-Embeddings for Improved Sentence Representations](https://arxiv.org/abs/1804.07983)|19|[Pytorch](https://github.com/facebookresearch/DME )|[DME/CDME](https://github.com/facebookresearch/DME#pre-trained-models )|
|2018/04|[Representation Tradeoffs for Hyperbolic Embeddings](https://arxiv.org/abs/1804.03329)|38|[Pytorch](https://github.com/HazyResearch/hyperbolics )|[h-MDS](https://github.com/HazyResearch/hyperbolics )|
|2018/05|[Analogical Reasoning on Chinese Morphological and Semantic Relations](https://arxiv.org/abs/1805.06504)|38|-|[ChineseWordVectors](https://github.com/Embedding/Chinese-Word-Vectors )|
|2018/06|[Probabilistic FastText for Multi-Sense Word Embeddings](https://arxiv.org/abs/1806.02901)|13|[C++](https://github.com/benathi/multisense-prob-fasttext )|[Probabilistic FastText](https://github.com/benathi/multisense-prob-fasttext#3-loading-and-analyzing-pre-trained-models )|
|2018/09|[Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks](https://arxiv.org/abs/1809.04283)|1|[TF](https://github.com/malllabiisc/WordGCN )|[SynGCN](https://drive.google.com/open?id=17wgNSMkyQwVHeHipk_Mp3y2Q0Kvhu6Mm )|
|2018/09|[FRAGE: Frequency-Agnostic Word Representation](https://arxiv.org/abs/1809.06858)|30|[Pytorch](https://github.com/ChengyueGongR/Frequency-Agnostic )|-|
|2018/12|[Wikipedia2Vec: An Optimized Tool for LearningEmbeddings of Words and Entities from Wikipedia](https://arxiv.org/abs/1812.06280)|3|[Cython](https://github.com/wikipedia2vec/wikipedia2vec )|[Wikipedia2Vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/ )|
|2018/??|[cw2vec: Learning Chinese Word Embeddings with Stroke n-gram Information](http://www.statnlp.org/wp-content/uploads/papers/2018/cw2vec/cw2vec.pdf)|18|[C++](https://github.com/bamtercelboo/cw2vec )|-|
|2019/02|[VCWE: Visual Character-Enhanced Word Embeddings](https://arxiv.org/abs/1902.08795)|0|[Pytorch](https://github.com/HSLCY/VCWE )|[VCWE](https://github.com/HSLCY/VCWE/blob/master/embedding/zh_wiki_VCWE_ep50.txt )|
|2019/05|[Learning Cross-lingual Embeddings from Twitter via Distant Supervision](https://arxiv.org/abs/1905.07358)|1|[Text](https://github.com/pedrada88/crossembeddings-twitter )|-|
|2019/08|[ViCo: Word Embeddings from Visual Co-occurrences](https://arxiv.org/abs/1908.08527)|0|[Pytorch](https://github.com/BigRedT/vico/ )|[ViCo](https://github.com/BigRedT/vico/#just-give-me-pretrained-vico )|
|2019/08|[An Unsupervised Character-Aware Neural Approach to Word and Context Representation Learning](https://arxiv.org/abs/1908.01819)|2|[TF](https://github.com/GiuseppeMarra/char-word-embeddings )|-|
|2019/11|[Spherical Text Embedding](https://arxiv.org/abs/1911.01196)|0|[C](https://github.com/yumeng5/Spherical-Text-Embedding )|-|
|2019/??|[Unsupervised word embeddings capture latent knowledge from materials science literature](https://www.nature.com/articles/s41586-019-1335-8)|6|[Gensim](https://github.com/materialsintelligence/mat2vec )|-|

## OOV Handling

* Drop OOV words!
* One OOV vector(unk vector)
* Use subword models(ngram, bpe, char)
* [ALaCarte](https://github.com/NLPrinceton/ALaCarte): [A La Carte Embedding: Cheap but Effective Induction of Semantic Feature Vectors](http://aclweb.org/anthology/P18-1002)
* [Mimick](https://github.com/yuvalpinter/Mimick): [Mimicking Word Embeddings using Subword RNNs](http://www.aclweb.org/anthology/D17-1010)
* [CompactReconstruction](https://github.com/losyer/compact_reconstruction): [Subword-based Compact Reconstruction of Word Embeddings](https://www.aclweb.org/anthology/N19-1353)

## Contextualized Word Embeddings

* Note: all the unofficial models can load the official pretrained models

|date|paper|citation count|code|pretrained models|
|:---:|:---:|:---:|:---:|:---:|
|-|[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)|N/A|[TF](https://github.com/openai/gpt-2 )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers )<br>[Keras](https://github.com/CyberZHG/keras-gpt-2 )|[GPT-2](https://github.com/openai/gpt-2 )|
|2017/08|[Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)|276|[Pytorch](https://github.com/salesforce/cove )<br>[Keras](https://github.com/rgsachin/CoVe )|[CoVe](https://github.com/salesforce/cove )|
|2018/01|[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)|71|[Pytorch](https://github.com/fastai/fastai/tree/ulmfit_v1 )|ULMFit([English](https://docs.fast.ai/text.html#Fine-tuning-a-language-model), [Zoo](https://forums.fast.ai/t/language-model-zoo-gorilla/14623/1))|
|2018/02|[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)|999+|[Pytorch](https://github.com/allenai/allennlp )<br>[TF](https://github.com/allenai/bilm-tf )|ELMO([AllenNLP](https://allennlp.org/elmo), [TF-Hub](https://tfhub.dev/google/elmo/2))|
|2018/04|[Efficient Contextualized Representation:Language Model Pruning for Sequence Labeling](https://arxiv.org/abs/1804.07827)|11|[Pytorch](https://github.com/LiyuanLucasLiu/LD-Net )|[LD-Net](https://github.com/LiyuanLucasLiu/LD-Net#language-models )|
|2018/07|[Towards Better UD Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation](https://arxiv.org/abs/1807.03121)|43|[Pytorch](https://github.com/HIT-SCIR/ELMoForManyLangs )|[ELMo](https://github.com/HIT-SCIR/ELMoForManyLangs#downloads )|
|2018/08|[Direct Output Connection for a High-Rank Language Model](https://arxiv.org/abs/1808.10143)|11|[Pytorch](https://github.com/nttcslab-nlp/doc_lm )|[DOC](https://drive.google.com/open?id=1ug-6ISrXHEGcWTk5KIw8Ojdjuww-i-Ci )|
|2018/10|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)|999+|[TF](https://github.com/google-research/bert )<br>[Keras](https://github.com/Separius/BERT-keras )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers )<br>[MXNet](https://github.com/imgarylai/bert-embedding )<br>[PaddlePaddle](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE )<br>[TF](https://github.com/hanxiao/bert-as-service/ )<br>[Keras](https://github.com/CyberZHG/keras-bert )|BERT([BERT](https://github.com/google-research/bert#pre-trained-models), [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE), [KoBERT](https://github.com/SKTBrain/KoBERT))|
|2018/??|[Contextual String Embeddings for Sequence Labeling]()|109|[Pytorch](https://github.com/zalandoresearch/flair )|[Flair](https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py#L407 )|
|2018/??|[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)|473|[TF](https://github.com/openai/finetune-transformer-lm )<br>[Keras](https://github.com/Separius/BERT-keras )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers )|[GPT](https://github.com/openai/finetune-transformer-lm )|
|2019/01|[Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)|108|[Pytorch](https://github.com/facebookresearch/XLM )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers )|[XLM](https://github.com/facebookresearch/XLM#pretrained-models )|
|2019/01|[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)|127|[TF](https://github.com/kimiyoung/transformer-xl/tree/master/tf )<br>[Pytorch](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers )|[Transformer-XL](https://github.com/kimiyoung/transformer-xl/tree/master/tf )|
|2019/01|[BioBERT: pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746)|70|[TF](https://github.com/dmis-lab/biobert )|[BioBERT](https://github.com/naver/biobert-pretrained )|
|2019/01|[Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/1901.11504)|79|[Pytorch](https://github.com/namisan/mt-dnn )|[MT-DNN](https://github.com/namisan/mt-dnn/blob/master/download.sh )|
|2019/02|[Efficient Contextual Representation Learning Without Softmax Layer](https://arxiv.org/abs/1902.11269)|1|[Pytorch](https://github.com/uclanlp/ELMO-C )|-|
|2019/03|[SciBERT: Pretrained Contextualized Embeddings for Scientific Text](https://arxiv.org/abs/1903.10676)|27|[Pytorch, TF](https://github.com/allenai/scibert )|[SciBERT](https://github.com/allenai/scibert#downloading-trained-models )|
|2019/04|[Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323)|16|[Text](https://github.com/EmilyAlsentzer/clinicalBERT )|[clinicalBERT](https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=0 )|
|2019/04|[ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342)|8|[Pytorch](https://github.com/kexinhuang12345/clinicalBERT )|[ClinicalBERT](https://drive.google.com/file/d/1t8L9w-r88Q5-sfC993x2Tjt1pu--A900/view )|
|2019/05|[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129)|17|[Pytorch](https://github.com/thunlp/ERNIE )|[ERNIE](https://drive.google.com/open?id=1m673-YB-4j1ISNDlk5oZjpPF2El7vn6f )|
|2019/05|[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)|21|[Pytorch](https://github.com/microsoft/unilm )|[UniLM](https://github.com/microsoft/unilm#pre-trained-models )|
|2019/05|[HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://arxiv.org/abs/1905.06566)|4||-|
|2019/06|[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)|112|[TF](https://github.com/zihangdai/xlnet )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers )|[XLNet](https://github.com/zihangdai/xlnet#released-models )|
|2019/06|[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)|3|[Pytorch, TF](https://github.com/ymcui/Chinese-BERT-wwm )|[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm#pytorch%E7%89%88%E6%9C%AC%E8%AF%B7%E4%BD%BF%E7%94%A8-%E7%9A%84pytorch-bert--06%E5%85%B6%E4%BB%96%E7%89%88%E6%9C%AC%E8%AF%B7%E8%87%AA%E8%A1%8C%E8%BD%AC%E6%8D%A2 )|
|2019/07|[ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/abs/1907.12412)|1|[PaddlePaddle](https://github.com/PaddlePaddle/ERNIE )|[ERNIE 2.0](https://github.com/PaddlePaddle/ERNIE#models )|
|2019/07|[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)|12|[Pytorch](https://github.com/facebookresearch/SpanBERT )|[SpanBERT](https://github.com/facebookresearch/SpanBERT#pre-trained-models )|
|2019/07|[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)|67|[Pytorch](https://github.com/pytorch/fairseq )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers )|[RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md#pre-trained-models )|
|2019/09|[MultiFiT: Efficient Multi-lingual Language Model Fine-tuning](https://arxiv.org/abs/1909.04761)|0|[Pytorch](https://github.com/n-waves/ulmfit-multilingual )|-|
|2019/09|[Extreme Language Model Compression with Optimal Subwords and Shared Projections](https://arxiv.org/abs/1909.11687)|0||-|
|2019/09|[UNITER: Learning UNiversal Image-TExt Representations](https://arxiv.org/abs/1909.11740)|0||-|
|2019/09|[MULE: Multimodal Universal Language Embedding](https://arxiv.org/abs/1909.03493)|0||-|
|2019/09|[Knowledge Enhanced Contextual Word Representations](https://arxiv.org/abs/1909.04164)|0||-|
|2019/09|[K-BERT: Enabling Language Representation with Knowledge Graph](https://arxiv.org/abs/1909.07606)|0||-|
|2019/09|[Subword ELMo](https://arxiv.org/abs/1909.08357)|0|[Pytorch](https://github.com/Jiangtong-Li/Subword-ELMo/ )|-|
|2019/09|[Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks](https://arxiv.org/abs/1909.00964)|1||-|
|2019/09|[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)|1||-|
|2019/09|[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)|3|[TF](https://github.com/brightmart/albert_zh )|-|
|2019/10|[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)|0|[TF](https://github.com/google-research/text-to-text-transfer-transformer )|[T5](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints )|
|2019/10|[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)|4|[Pytorch, TF2.0](https://github.com/huggingface/transformers )|[DistilBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation )|
|2019/11|[CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894)|0|-|[CamemBERT](https://camembert-model.fr/#download )|
|2019/11|[ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations](https://arxiv.org/abs/1911.00720)|0|[Pytorch](https://github.com/sinovation/ZEN )|-|

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

|date|paper|citation count|code|model_name|
|:---:|:---:|:---:|:---:|:---:|
|-|[Incremental Domain Adaptation for Neural Machine Translation in Low-Resource Settings](https://www.aclweb.org/anthology/W19-4601)|N/A|[Python](https://github.com/DFKI-Interactive-Machine-Learning/AraSIF )|AraSIF|
|2014/05|[Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053)|999+|[Pytorch](https://github.com/inejc/paragraph-vectors )<br>[Python](https://github.com/jhlau/doc2vec )|Doc2Vec|
|2014/11|[Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](https://arxiv.org/abs/1411.2539)|499|[Theano](https://github.com/ryankiros/visual-semantic-embedding )<br>[Pytorch](https://github.com/linxd5/VSE_Pytorch )|VSE|
|2015/06|[Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](https://arxiv.org/abs/1506.06724)|312|[Theano](https://github.com/ryankiros/skip-thoughts )<br>[TF](https://github.com/tensorflow/models/tree/master/research/skip_thoughts )<br>[Pytorch, Torch](https://github.com/Cadene/skip-thoughts.torch )|SkipThought|
|2015/11|[Order-Embeddings of Images and Language](https://arxiv.org/abs/1511.06361)|200|[Theano](https://github.com/ivendrov/order-embedding )|order-embedding|
|2015/11|[Towards Universal Paraphrastic Sentence Embeddings](https://arxiv.org/abs/1511.08198)|261|[Theano](https://github.com/jwieting/iclr2016 )|ParagramPhrase|
|2015/??|[From Word Embeddings to Document Distances](http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf)|536|[C, Python](https://github.com/mkusner/wmd )|Word Mover's Distance|
|2016/02|[Learning Distributed Representations of Sentences from Unlabelled Data](https://arxiv.org/abs/1602.03483)|250|[Python](https://github.com/fh295/SentenceRepresentation )|FastSent|
|2016/07|[Charagram: Embedding Words and Sentences via Character n-grams](https://arxiv.org/abs/1607.02789)|86|[Theano](https://github.com/jwieting/charagram )|Charagram|
|2016/11|[Learning Generic Sentence Representations Using Convolutional Neural Networks](https://arxiv.org/abs/1611.07897)|46|[Theano](https://github.com/zhegan27/ConvSent )|ConvSent|
|2017/03|[Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features](https://arxiv.org/abs/1703.02507)|148|[C++](https://github.com/epfml/sent2vec )|Sent2Vec|
|2017/04|[Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444)|177|[TF](https://github.com/openai/generating-reviews-discovering-sentiment )<br>[Pytorch](https://github.com/guillitte/pytorch-sentiment-neuron )<br>[Pytorch](https://github.com/NVIDIA/sentiment-discovery )|Sentiment Neuron|
|2017/05|[Revisiting Recurrent Networks for Paraphrastic Sentence Embeddings](https://arxiv.org/abs/1705.00364)|40|[Theano](https://github.com/jwieting/acl2017 )|GRAN|
|2017/05|[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364)|518|[Pytorch](https://github.com/facebookresearch/InferSent )|InferSent|
|2017/07|[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612)|71|[Pytorch](https://github.com/fartashf/vsepp )|VSE++|
|2017/08|[Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](https://arxiv.org/abs/1708.00524)|167|[Keras](https://github.com/bfelbo/DeepMoji )<br>[Pytorch](https://github.com/huggingface/torchMoji )|DeepMoji|
|2017/09|[StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856)|57|[C++](https://github.com/facebookresearch/StarSpace )|StarSpace|
|2017/10|[DisSent: Learning Sentence Representations from Explicit Discourse Relations](https://arxiv.org/abs/1710.04334)|38|[Pytorch](https://github.com/windweller/DisExtract )|DisSent|
|2017/11|[Dual-Path Convolutional Image-Text Embedding with Instance Loss](https://arxiv.org/abs/1711.05535)|22|[Matlab](https://github.com/layumi/Image-Text-Embedding )|Image-Text-Embedding|
|2017/11|[Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations](https://arxiv.org/abs/1711.05732)|48|[Theano](https://github.com/jwieting/para-nmt-50m )|para-nmt|
|2018/03|[Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)|127|[TF-Hub](https://tfhub.dev/google/universal-sentence-encoder-large/2 )|USE|
|2018/03|[An efficient framework for learning sentence representations](https://arxiv.org/abs/1803.02893)|69|[TF](https://github.com/lajanugen/S2V )|Quick-Thought|
|2018/04|[End-Task Oriented Textual Entailment via Deep Explorations of Inter-Sentence Interactions](https://arxiv.org/abs/1804.08813)|7|[Theano](https://github.com/yinwenpeng/SciTail )|DEISTE|
|2018/04|[Learning general purpose distributed sentence representations via large scale multi-task learning](https://arxiv.org/abs/1804.00079)|98|[Pytorch](https://github.com/Maluuba/gensen )|GenSen|
|2018/06|[Embedding Text in Hyperbolic Spaces](https://arxiv.org/abs/1806.04313)|19|[TF](https://github.com/brain-research/hyperbolictext )|HyperText|
|2018/07|[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)|95|[Keras](https://github.com/davidtellez/contrastive-predictive-coding )|CPC|
|2018/08|[Context Mover’s Distance & Barycenters: Optimal transport of contexts for building representations](https://arxiv.org/abs/1808.09663)|2|[Python](https://github.com/context-mover/context-mover-distance-and-barycenters )|CMD|
|2018/09|[Learning Universal Sentence Representations with Mean-Max Attention Autoencoder](https://arxiv.org/abs/1809.06590)|3|[TF](https://github.com/Zminghua/SentEncoding )|Mean-MaxAAE|
|2018/10|[Improving Sentence Representations with Consensus Maximisation](https://arxiv.org/abs/1810.01064)|1|-|Multi-view|
|2018/10|[BioSentVec: creating sentence embeddings for biomedical texts](https://arxiv.org/abs/1810.09302)|13|[Python](https://github.com/ncbi-nlp/BioSentVec )|BioSentVec|
|2018/10|[Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model](https://arxiv.org/abs/1810.12836)|5|[TF-Hub](https://tfhub.dev/s?q=universal-sentence-encoder-xling )|USE-xling|
|2018/11|[Word Mover's Embedding: From Word2Vec to Document Embedding](https://arxiv.org/abs/1811.01713)|13|[C, Python](https://github.com/IBM/WordMoversEmbeddings )|WordMoversEmbeddings|
|2018/11|[A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks](https://arxiv.org/abs/1811.06031)|23|[Pytorch](https://github.com/huggingface/hmtl )|HMTL|
|2018/12|[Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464)|44|[Pytorch](https://github.com/facebookresearch/LASER )|LASER|
|2018/??|[Convolutional Neural Network for Universal Sentence Embeddings](https://pdfs.semanticscholar.org/d827/32de6336dd6443ff33cccbb92ced0196ecc1.pdf)|0|[Theano](https://github.com/XiaoqiJiao/COLING2018 )|CSE|
|2019/01|[No Training Required: Exploring Random Encoders for Sentence Classification](https://arxiv.org/abs/1901.10444)|14|[Pytorch](https://github.com/facebookresearch/randsent )|randsent|
|2019/02|[CBOW Is Not All You Need: Combining CBOW with the Compositional Matrix Space Model](https://arxiv.org/abs/1902.06423)|0|[Pytorch](https://github.com/florianmai/word2mat )|CMOW|
|2019/07|[GLOSS: Generative Latent Optimization of Sentence Representations](https://arxiv.org/abs/1907.06385)|0|-|GLOSS|
|2019/07|[Multilingual Universal Sentence Encoder](https://arxiv.org/abs/1907.04307)|1|[TF-Hub](https://tfhub.dev/google/universal-sentence-encoder-multilingual/1 )|MultilingualUSE|
|2019/08|[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)|1|[Pytorch](https://github.com/UKPLab/sentence-transformers )|Sentence-BERT|

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
