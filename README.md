# awesome-sentence-embedding [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

[![Build Status](https://travis-ci.com/Separius/awesome-sentence-embedding.svg?branch=master)](https://travis-ci.com/Separius/awesome-sentence-embedding)
[![GitHub - LICENSE](https://img.shields.io/github/license/Separius/awesome-sentence-embedding.svg?style=flat)](./LICENSE)

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
|-|[Dict2vec : Learning Word Embeddings using Lexical Dictionaries](http://aclweb.org/anthology/D17-1024)|N/A|[C++](https://github.com/tca19/dict2vec ) ![](https://img.shields.io/github/stars/tca19/dict2vec.svg?style=social )|[Dict2vec](https://github.com/tca19/dict2vec#download-pre-trained-vectors )|
|-|[WebVectors: A Toolkit for Building Web Interfaces for Vector Semantic Models](https://rusvectores.org/static/data/webvectors_aist.pdf)|N/A|-|[RusVectōrēs](http://rusvectores.org/en/models/ )|
|-|[Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings](https://ai.tencent.com/ailab/media/publications/naacl2018/directional_skip-gram.pdf)|N/A|-|[ChineseEmbedding](https://ai.tencent.com/ailab/nlp/en/embedding.html )|
|-|[Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics](http://www.aclweb.org/anthology/D17-1023)|N/A|[C](https://github.com/zhezhaoa/ngram2vec ) ![](https://img.shields.io/github/stars/zhezhaoa/ngram2vec.svg?style=social )|-|
|-|[Joint Embeddings of Chinese Words, Characters, and Fine-grained Subcharacter Components](https://www.cse.ust.hk/~yqsong/papers/2017-EMNLP-ChineseEmbedding.pdf)|N/A|[C](https://github.com/hkust-knowcomp/jwe ) ![](https://img.shields.io/github/stars/hkust-knowcomp/jwe.svg?style=social )|-|
|2013/01|[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)|999+|[C](https://github.com/tmikolov/word2vec ) ![](https://img.shields.io/github/stars/tmikolov/word2vec.svg?style=social )|[Word2Vec](https://code.google.com/archive/p/word2vec/ )|
|2014/12|[Word Representations via Gaussian Embedding](https://arxiv.org/abs/1412.6623)|201|[Cython](https://github.com/seomoz/word2gauss ) ![](https://img.shields.io/github/stars/seomoz/word2gauss.svg?style=social )|-|
|2014/??|[A Probabilistic Model for Learning Multi-Prototype Word Embeddings](http://www.aclweb.org/anthology/C14-1016)|123|[DMTK](https://github.com/Microsoft/distributed_skipgram_mixture ) ![](https://img.shields.io/github/stars/Microsoft/distributed_skipgram_mixture.svg?style=social )|-|
|2014/??|[Dependency-Based Word Embeddings](http://www.aclweb.org/anthology/P14-2050)|669|[C++](https://bitbucket.org/yoavgo/word2vecf/src/default/ )|[word2vecf](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/ )|
|2014/??|[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)|999+|[C](https://github.com/stanfordnlp/GloVe ) ![](https://img.shields.io/github/stars/stanfordnlp/GloVe.svg?style=social )|[GloVe](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors )|
|2015/06|[Sparse Overcomplete Word Vector Representations](https://arxiv.org/abs/1506.02004)|118|[C++](https://github.com/mfaruqui/sparse-coding ) ![](https://img.shields.io/github/stars/mfaruqui/sparse-coding.svg?style=social )|-|
|2015/06|[From Paraphrase Database to Compositional Paraphrase Model and Back](https://arxiv.org/abs/1506.03487)|3|[Theano](https://github.com/jwieting/paragram-word ) ![](https://img.shields.io/github/stars/jwieting/paragram-word.svg?style=social )|[PARAGRAM](http://ttic.uchicago.edu/~wieting/paragram-word-demo.zip )|
|2015/06|[Non-distributional Word Vector Representations](https://arxiv.org/abs/1506.05230)|64|[Python](https://github.com/mfaruqui/non-distributional ) ![](https://img.shields.io/github/stars/mfaruqui/non-distributional.svg?style=social )|[WordFeat](https://github.com/mfaruqui/non-distributional/blob/master/binary-vectors.txt.gz )|
|2015/??|[Joint Learning of Character and Word Embeddings](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/download/11000/10834)|178|[C](https://github.com/Leonard-Xu/CWE ) ![](https://img.shields.io/github/stars/Leonard-Xu/CWE.svg?style=social )|-|
|2015/??|[SensEmbed: Learning Sense Embeddings for Word and Relational Similarity](http://www.aclweb.org/anthology/P/P15/P15-1010.pdf)|237|-|[SensEmbed](http://lcl.uniroma1.it/sensembed/sensembed_vectors.gz )|
|2015/??|[Topical Word Embeddings](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9314/9535)|267|[Cython](https://github.com/largelymfs/topical_word_embeddings ) ![](https://img.shields.io/github/stars/largelymfs/topical_word_embeddings.svg?style=social )|[]( )|
|2016/02|[Swivel: Improving Embeddings by Noticing What's Missing](https://arxiv.org/abs/1602.02215)|58|[TF](https://github.com/tensorflow/models/tree/master/research/swivel ) ![](https://img.shields.io/github/stars/tensorflow/models.svg?style=social )|-|
|2016/03|[Counter-fitting Word Vectors to Linguistic Constraints](https://arxiv.org/abs/1603.00892)|198|[Python](https://github.com/nmrksic/counter-fitting ) ![](https://img.shields.io/github/stars/nmrksic/counter-fitting.svg?style=social )|[counter-fitting](http://mi.eng.cam.ac.uk/~nm480/counter-fitted-vectors.txt.zip )(broken)|
|2016/05|[Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/abs/1605.02019)|73|[Chainer](https://github.com/cemoody/lda2vec ) ![](https://img.shields.io/github/stars/cemoody/lda2vec.svg?style=social )|-|
|2016/06|[Siamese CBOW: Optimizing Word Embeddings for Sentence Representations](https://arxiv.org/abs/1606.04640)|155|[Theano](https://bitbucket.org/TomKenter/siamese-cbow/src/master/ )|[Siamese CBOW](https://bitbucket.org/TomKenter/siamese-cbow/src/master/ )|
|2016/06|[Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations](https://arxiv.org/abs/1606.00819)|54|[Go](https://github.com/alexandres/lexvec ) ![](https://img.shields.io/github/stars/alexandres/lexvec.svg?style=social )|[lexvec](https://github.com/alexandres/lexvec#pre-trained-vectors )|
|2016/07|[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)|999+|[C++](https://github.com/facebookresearch/fastText ) ![](https://img.shields.io/github/stars/facebookresearch/fastText.svg?style=social )|[fastText](https://fasttext.cc/docs/en/english-vectors.html )|
|2016/08|[Morphological Priors for Probabilistic Neural Word Embeddings](https://arxiv.org/abs/1608.01056)|29|[Theano](https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings ) ![](https://img.shields.io/github/stars/rguthrie3/MorphologicalPriorsForWordEmbeddings.svg?style=social )|-|
|2016/11|[A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks](https://arxiv.org/abs/1611.01587)|306|[C++](https://github.com/hassyGo/charNgram2vec ) ![](https://img.shields.io/github/stars/hassyGo/charNgram2vec.svg?style=social )|[charNgram2vec](https://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz )|
|2016/12|[ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://arxiv.org/abs/1612.03975)|438|[Python](https://github.com/commonsense/conceptnet-numberbatch ) ![](https://img.shields.io/github/stars/commonsense/conceptnet-numberbatch.svg?style=social )|[Numberbatch](https://github.com/commonsense/conceptnet-numberbatch#downloads )|
|2016/??|[Learning Word Meta-Embeddings](http://www.aclweb.org/anthology/P16-1128)|45|-|[Meta-Emb](http://cistern.cis.lmu.de/meta-emb/ )(broken)|
|2017/02|[Offline bilingual word vectors, orthogonal transformations and the inverted softmax](https://arxiv.org/abs/1702.03859)|291|[Python](https://github.com/Babylonpartners/fastText_multilingual ) ![](https://img.shields.io/github/stars/Babylonpartners/fastText_multilingual.svg?style=social )|-|
|2017/04|[Multimodal Word Distributions](https://arxiv.org/abs/1704.08424)|50|[TF](https://github.com/benathi/word2gm ) ![](https://img.shields.io/github/stars/benathi/word2gm.svg?style=social )|[word2gm](https://github.com/benathi/word2gm#trained-model )|
|2017/05|[Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039)|327|[Pytorch](https://github.com/facebookresearch/poincare-embeddings ) ![](https://img.shields.io/github/stars/facebookresearch/poincare-embeddings.svg?style=social )|-|
|2017/06|[Context encoders as a simple but powerful extension of word2vec](https://arxiv.org/abs/1706.02496)|11|[Python](https://github.com/cod3licious/conec ) ![](https://img.shields.io/github/stars/cod3licious/conec.svg?style=social )|-|
|2017/06|[Semantic Specialisation of Distributional Word Vector Spaces using Monolingual and Cross-Lingual Constraints](https://arxiv.org/abs/1706.00374)|81|[TF](https://github.com/nmrksic/attract-repel ) ![](https://img.shields.io/github/stars/nmrksic/attract-repel.svg?style=social )|[Attract-Repel](https://github.com/nmrksic/attract-repel#available-word-vector-spaces )|
|2017/08|[Learning Chinese Word Representations From Glyphs Of Characters](https://arxiv.org/abs/1708.04755)|38|[C](https://github.com/ray1007/gwe ) ![](https://img.shields.io/github/stars/ray1007/gwe.svg?style=social )|-|
|2017/08|[Making Sense of Word Embeddings](https://arxiv.org/abs/1708.03390)|84|[Python](https://github.com/uhh-lt/sensegram ) ![](https://img.shields.io/github/stars/uhh-lt/sensegram.svg?style=social )|[sensegram](http://ltdata1.informatik.uni-hamburg.de/sensegram/ )|
|2017/09|[Hash Embeddings for Efficient Word Representations](https://arxiv.org/abs/1709.03933)|19|[Keras](https://github.com/dsv77/hashembedding ) ![](https://img.shields.io/github/stars/dsv77/hashembedding.svg?style=social )|-|
|2017/10|[BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages](https://arxiv.org/abs/1710.02187)|65|[Gensim](https://github.com/bheinzerling/bpemb ) ![](https://img.shields.io/github/stars/bheinzerling/bpemb.svg?style=social )|[BPEmb](https://github.com/bheinzerling/bpemb#downloads-for-each-language )|
|2017/11|[SPINE: SParse Interpretable Neural Embeddings](https://arxiv.org/abs/1711.08792)|41|[Pytorch](https://github.com/harsh19/SPINE ) ![](https://img.shields.io/github/stars/harsh19/SPINE.svg?style=social )|[SPINE](https://drive.google.com/drive/folders/1ksVcWDADmnp0Cl5kezjHqTg3Jnh8q031?usp=sharing )|
|2017/??|[AraVec: A set of Arabic Word Embedding Models for use in Arabic NLP](https://www.researchgate.net/publication/319880027_AraVec_A_set_of_Arabic_Word_Embedding_Models_for_use_in_Arabic_NLP)|122|[Gensim](https://github.com/bakrianoo/aravec ) ![](https://img.shields.io/github/stars/bakrianoo/aravec.svg?style=social )|[AraVec](https://github.com/bakrianoo/aravec#n-grams-models-1 )|
|2018/04|[Representation Tradeoffs for Hyperbolic Embeddings](https://arxiv.org/abs/1804.03329)|100|[Pytorch](https://github.com/HazyResearch/hyperbolics ) ![](https://img.shields.io/github/stars/HazyResearch/hyperbolics.svg?style=social )|[h-MDS](https://github.com/HazyResearch/hyperbolics )|
|2018/04|[Dynamic Meta-Embeddings for Improved Sentence Representations](https://arxiv.org/abs/1804.07983)|42|[Pytorch](https://github.com/facebookresearch/DME ) ![](https://img.shields.io/github/stars/facebookresearch/DME.svg?style=social )|[DME/CDME](https://github.com/facebookresearch/DME#pre-trained-models )|
|2018/05|[Analogical Reasoning on Chinese Morphological and Semantic Relations](https://arxiv.org/abs/1805.06504)|95|-|[ChineseWordVectors](https://github.com/Embedding/Chinese-Word-Vectors )|
|2018/06|[Probabilistic FastText for Multi-Sense Word Embeddings](https://arxiv.org/abs/1806.02901)|30|[C++](https://github.com/benathi/multisense-prob-fasttext ) ![](https://img.shields.io/github/stars/benathi/multisense-prob-fasttext.svg?style=social )|[Probabilistic FastText](https://github.com/benathi/multisense-prob-fasttext#3-loading-and-analyzing-pre-trained-models )|
|2018/09|[Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks](https://arxiv.org/abs/1809.04283)|1|[TF](https://github.com/malllabiisc/WordGCN ) ![](https://img.shields.io/github/stars/malllabiisc/WordGCN.svg?style=social )|[SynGCN](https://drive.google.com/open?id=17wgNSMkyQwVHeHipk_Mp3y2Q0Kvhu6Mm )|
|2018/09|[FRAGE: Frequency-Agnostic Word Representation](https://arxiv.org/abs/1809.06858)|50|[Pytorch](https://github.com/ChengyueGongR/Frequency-Agnostic ) ![](https://img.shields.io/github/stars/ChengyueGongR/Frequency-Agnostic.svg?style=social )|-|
|2018/12|[Wikipedia2Vec: An Optimized Tool for LearningEmbeddings of Words and Entities from Wikipedia](https://arxiv.org/abs/1812.06280)|22|[Cython](https://github.com/wikipedia2vec/wikipedia2vec ) ![](https://img.shields.io/github/stars/wikipedia2vec/wikipedia2vec.svg?style=social )|[Wikipedia2Vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/ )|
|2018/??|[cw2vec: Learning Chinese Word Embeddings with Stroke n-gram Information](http://www.statnlp.org/wp-content/uploads/papers/2018/cw2vec/cw2vec.pdf)|40|[C++](https://github.com/bamtercelboo/cw2vec ) ![](https://img.shields.io/github/stars/bamtercelboo/cw2vec.svg?style=social )|-|
|2019/02|[VCWE: Visual Character-Enhanced Word Embeddings](https://arxiv.org/abs/1902.08795)|3|[Pytorch](https://github.com/HSLCY/VCWE ) ![](https://img.shields.io/github/stars/HSLCY/VCWE.svg?style=social )|[VCWE](https://github.com/HSLCY/VCWE/blob/master/embedding/zh_wiki_VCWE_ep50.txt )|
|2019/05|[Learning Cross-lingual Embeddings from Twitter via Distant Supervision](https://arxiv.org/abs/1905.07358)|2|[Text](https://github.com/pedrada88/crossembeddings-twitter ) ![](https://img.shields.io/github/stars/pedrada88/crossembeddings-twitter.svg?style=social )|-|
|2019/08|[An Unsupervised Character-Aware Neural Approach to Word and Context Representation Learning](https://arxiv.org/abs/1908.01819)|2|[TF](https://github.com/GiuseppeMarra/char-word-embeddings ) ![](https://img.shields.io/github/stars/GiuseppeMarra/char-word-embeddings.svg?style=social )|-|
|2019/08|[ViCo: Word Embeddings from Visual Co-occurrences](https://arxiv.org/abs/1908.08527)|4|[Pytorch](https://github.com/BigRedT/vico/ ) ![](https://img.shields.io/github/stars/BigRedT/vico.svg?style=social )|[ViCo](https://github.com/BigRedT/vico/#just-give-me-pretrained-vico )|
|2019/11|[Spherical Text Embedding](https://arxiv.org/abs/1911.01196)|12|[C](https://github.com/yumeng5/Spherical-Text-Embedding ) ![](https://img.shields.io/github/stars/yumeng5/Spherical-Text-Embedding.svg?style=social )|-|
|2019/??|[Unsupervised word embeddings capture latent knowledge from materials science literature](https://www.nature.com/articles/s41586-019-1335-8)|88|[Gensim](https://github.com/materialsintelligence/mat2vec ) ![](https://img.shields.io/github/stars/materialsintelligence/mat2vec.svg?style=social )|-|

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
|-|[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)|N/A|[TF](https://github.com/openai/gpt-2 ) ![](https://img.shields.io/github/stars/openai/gpt-2.svg?style=social )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers ) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social )<br>[Keras](https://github.com/CyberZHG/keras-gpt-2 ) ![](https://img.shields.io/github/stars/CyberZHG/keras-gpt-2.svg?style=social )|GPT-2([117M](https://github.com/openai/gpt-2), [124M](https://github.com/openai/gpt-2), [345M](https://github.com/openai/gpt-2), [355M](https://github.com/openai/gpt-2), [774M](https://github.com/openai/gpt-2), [1558M](https://github.com/openai/gpt-2))|
|2017/08|[Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)|466|[Pytorch](https://github.com/salesforce/cove ) ![](https://img.shields.io/github/stars/salesforce/cove.svg?style=social )<br>[Keras](https://github.com/rgsachin/CoVe ) ![](https://img.shields.io/github/stars/rgsachin/CoVe.svg?style=social )|[CoVe](https://github.com/salesforce/cove )|
|2018/01|[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)|231|[Pytorch](https://github.com/fastai/fastai/tree/ulmfit_v1 ) ![](https://img.shields.io/github/stars/fastai/fastai.svg?style=social )|ULMFit([English](https://docs.fast.ai/text.html#Fine-tuning-a-language-model), [Zoo](https://forums.fast.ai/t/language-model-zoo-gorilla/14623/1))|
|2018/02|[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)|999+|[Pytorch](https://github.com/allenai/allennlp ) ![](https://img.shields.io/github/stars/allenai/allennlp.svg?style=social )<br>[TF](https://github.com/allenai/bilm-tf ) ![](https://img.shields.io/github/stars/allenai/bilm-tf.svg?style=social )|ELMO([AllenNLP](https://allennlp.org/elmo), [TF-Hub](https://tfhub.dev/google/elmo/2))|
|2018/04|[Efficient Contextualized Representation:Language Model Pruning for Sequence Labeling](https://arxiv.org/abs/1804.07827)|18|[Pytorch](https://github.com/LiyuanLucasLiu/LD-Net ) ![](https://img.shields.io/github/stars/LiyuanLucasLiu/LD-Net.svg?style=social )|[LD-Net](https://github.com/LiyuanLucasLiu/LD-Net#language-models )|
|2018/07|[Towards Better UD Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation](https://arxiv.org/abs/1807.03121)|96|[Pytorch](https://github.com/HIT-SCIR/ELMoForManyLangs ) ![](https://img.shields.io/github/stars/HIT-SCIR/ELMoForManyLangs.svg?style=social )|[ELMo](https://github.com/HIT-SCIR/ELMoForManyLangs#downloads )|
|2018/08|[Direct Output Connection for a High-Rank Language Model](https://arxiv.org/abs/1808.10143)|21|[Pytorch](https://github.com/nttcslab-nlp/doc_lm ) ![](https://img.shields.io/github/stars/nttcslab-nlp/doc_lm.svg?style=social )|[DOC](https://drive.google.com/open?id=1ug-6ISrXHEGcWTk5KIw8Ojdjuww-i-Ci )|
|2018/10|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)|999+|[TF](https://github.com/google-research/bert ) ![](https://img.shields.io/github/stars/google-research/bert.svg?style=social )<br>[Keras](https://github.com/Separius/BERT-keras ) ![](https://img.shields.io/github/stars/Separius/BERT-keras.svg?style=social )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers ) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social )<br>[MXNet](https://github.com/imgarylai/bert-embedding ) ![](https://img.shields.io/github/stars/imgarylai/bert-embedding.svg?style=social )<br>[PaddlePaddle](https://github.com/PaddlePaddle/ERNIE ) ![](https://img.shields.io/github/stars/PaddlePaddle/ERNIE.svg?style=social )<br>[TF](https://github.com/hanxiao/bert-as-service/ ) ![](https://img.shields.io/github/stars/hanxiao/bert-as-service.svg?style=social )<br>[Keras](https://github.com/CyberZHG/keras-bert ) ![](https://img.shields.io/github/stars/CyberZHG/keras-bert.svg?style=social )|BERT([BERT](https://github.com/google-research/bert#pre-trained-models), [ERNIE](https://github.com/PaddlePaddle/ERNIE), [KoBERT](https://github.com/SKTBrain/KoBERT))|
|2018/??|[Contextual String Embeddings for Sequence Labeling]()|322|[Pytorch](https://github.com/zalandoresearch/flair ) ![](https://img.shields.io/github/stars/zalandoresearch/flair.svg?style=social )|[Flair](https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py#L407 )|
|2018/??|[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)|999+|[TF](https://github.com/openai/finetune-transformer-lm ) ![](https://img.shields.io/github/stars/openai/finetune-transformer-lm.svg?style=social )<br>[Keras](https://github.com/Separius/BERT-keras ) ![](https://img.shields.io/github/stars/Separius/BERT-keras.svg?style=social )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers ) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social )|[GPT](https://github.com/openai/finetune-transformer-lm )|
|2019/01|[Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/1901.11504)|261|[Pytorch](https://github.com/namisan/mt-dnn ) ![](https://img.shields.io/github/stars/namisan/mt-dnn.svg?style=social )|[MT-DNN](https://github.com/namisan/mt-dnn/blob/master/download.sh )|
|2019/01|[BioBERT: pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746)|320|[TF](https://github.com/dmis-lab/biobert ) ![](https://img.shields.io/github/stars/dmis-lab/biobert.svg?style=social )|[BioBERT](https://github.com/naver/biobert-pretrained )|
|2019/01|[Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)|377|[Pytorch](https://github.com/facebookresearch/XLM ) ![](https://img.shields.io/github/stars/facebookresearch/XLM.svg?style=social )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers ) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social )|[XLM](https://github.com/facebookresearch/XLM#pretrained-models )|
|2019/01|[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)|501|[TF](https://github.com/kimiyoung/transformer-xl/tree/master/tf ) ![](https://img.shields.io/github/stars/kimiyoung/transformer-xl.svg?style=social )<br>[Pytorch](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch ) ![](https://img.shields.io/github/stars/kimiyoung/transformer-xl.svg?style=social )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers ) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social )|[Transformer-XL](https://github.com/kimiyoung/transformer-xl/tree/master/tf )|
|2019/02|[Efficient Contextual Representation Learning Without Softmax Layer](https://arxiv.org/abs/1902.11269)|1|[Pytorch](https://github.com/uclanlp/ELMO-C ) ![](https://img.shields.io/github/stars/uclanlp/ELMO-C.svg?style=social )|-|
|2019/03|[SciBERT: Pretrained Contextualized Embeddings for Scientific Text](https://arxiv.org/abs/1903.10676)|88|[Pytorch, TF](https://github.com/allenai/scibert ) ![](https://img.shields.io/github/stars/allenai/scibert.svg?style=social )|[SciBERT](https://github.com/allenai/scibert#downloading-trained-models )|
|2019/04|[Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323)|107|[Text](https://github.com/EmilyAlsentzer/clinicalBERT ) ![](https://img.shields.io/github/stars/EmilyAlsentzer/clinicalBERT.svg?style=social )|[clinicalBERT](https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=0 )|
|2019/04|[ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342)|39|[Pytorch](https://github.com/kexinhuang12345/clinicalBERT ) ![](https://img.shields.io/github/stars/kexinhuang12345/clinicalBERT.svg?style=social )|[ClinicalBERT](https://drive.google.com/file/d/1t8L9w-r88Q5-sfC993x2Tjt1pu--A900/view )|
|2019/05|[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129)|100|[Pytorch](https://github.com/thunlp/ERNIE ) ![](https://img.shields.io/github/stars/thunlp/ERNIE.svg?style=social )|[ERNIE](https://drive.google.com/open?id=1m673-YB-4j1ISNDlk5oZjpPF2El7vn6f )|
|2019/05|[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)|152|[Pytorch](https://github.com/microsoft/unilm/tree/master/unilm-v1 ) ![](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social )|UniLMv1([unilm1-large-cased](https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin), [unilm1-base-cased](https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin))|
|2019/05|[HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://arxiv.org/abs/1905.06566)|33||-|
|2019/06|[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)|46|[Pytorch, TF](https://github.com/ymcui/Chinese-BERT-wwm ) ![](https://img.shields.io/github/stars/ymcui/Chinese-BERT-wwm.svg?style=social )|[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm#pytorch%E7%89%88%E6%9C%AC%E8%AF%B7%E4%BD%BF%E7%94%A8-%E7%9A%84pytorch-bert--06%E5%85%B6%E4%BB%96%E7%89%88%E6%9C%AC%E8%AF%B7%E8%87%AA%E8%A1%8C%E8%BD%AC%E6%8D%A2 )|
|2019/06|[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)|999+|[TF](https://github.com/zihangdai/xlnet ) ![](https://img.shields.io/github/stars/zihangdai/xlnet.svg?style=social )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers ) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social )|[XLNet](https://github.com/zihangdai/xlnet#released-models )|
|2019/07|[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)|139|[Pytorch](https://github.com/facebookresearch/SpanBERT ) ![](https://img.shields.io/github/stars/facebookresearch/SpanBERT.svg?style=social )|[SpanBERT](https://github.com/facebookresearch/SpanBERT#pre-trained-models )|
|2019/07|[ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/abs/1907.12412)|52|[PaddlePaddle](https://github.com/PaddlePaddle/ERNIE ) ![](https://img.shields.io/github/stars/PaddlePaddle/ERNIE.svg?style=social )|[ERNIE 2.0](https://github.com/PaddlePaddle/ERNIE#models )|
|2019/07|[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)|913|[Pytorch](https://github.com/pytorch/fairseq ) ![](https://img.shields.io/github/stars/pytorch/fairseq.svg?style=social )<br>[Pytorch, TF2.0](https://github.com/huggingface/transformers ) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social )|[RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md#pre-trained-models )|
|2019/09|[UNITER: Learning UNiversal Image-TExt Representations](https://arxiv.org/abs/1909.11740)|1||-|
|2019/09|[Subword ELMo](https://arxiv.org/abs/1909.08357)|1|[Pytorch](https://github.com/Jiangtong-Li/Subword-ELMo/ ) ![](https://img.shields.io/github/stars/Jiangtong-Li/Subword-ELMo.svg?style=social )|-|
|2019/09|[MultiFiT: Efficient Multi-lingual Language Model Fine-tuning](https://arxiv.org/abs/1909.04761)|15|[Pytorch](https://github.com/n-waves/ulmfit-multilingual ) ![](https://img.shields.io/github/stars/n-waves/ulmfit-multilingual.svg?style=social )|-|
|2019/09|[Extreme Language Model Compression with Optimal Subwords and Shared Projections](https://arxiv.org/abs/1909.11687)|17||-|
|2019/09|[K-BERT: Enabling Language Representation with Knowledge Graph](https://arxiv.org/abs/1909.07606)|19||-|
|2019/09|[MULE: Multimodal Universal Language Embedding](https://arxiv.org/abs/1909.03493)|2||-|
|2019/09|[Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks](https://arxiv.org/abs/1909.00964)|24||-|
|2019/09|[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)|359|[TF](https://github.com/brightmart/albert_zh ) ![](https://img.shields.io/github/stars/brightmart/albert_zh.svg?style=social )|-|
|2019/09|[Knowledge Enhanced Contextual Word Representations](https://arxiv.org/abs/1909.04164)|49||-|
|2019/09|[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)|62||-|
|2019/09|[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)|74|[Pytorch](https://github.com/NVIDIA/Megatron-LM ) ![](https://img.shields.io/github/stars/NVIDIA/Megatron-LM.svg?style=social )|Megatron-LM([BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m), [GPT-2-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_lm_345m))|
|2019/10|[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)|104|[Pytorch](https://github.com/pytorch/fairseq/tree/master/examples/bart ) ![](https://img.shields.io/github/stars/pytorch/fairseq.svg?style=social )|BART([bart.base](https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz), [bart.large](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz), [bart.large.mnli](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz), [bart.large.cnn](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz), [bart.large.xsum](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz))|
|2019/10|[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)|184|[Pytorch, TF2.0](https://github.com/huggingface/transformers ) ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social )|[DistilBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation )|
|2019/10|[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)|304|[TF](https://github.com/google-research/text-to-text-transfer-transformer ) ![](https://img.shields.io/github/stars/google-research/text-to-text-transfer-transformer.svg?style=social )|[T5](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints )|
|2019/11|[CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894)|36|-|[CamemBERT](https://camembert-model.fr/#download )|
|2019/11|[ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations](https://arxiv.org/abs/1911.00720)|4|[Pytorch](https://github.com/sinovation/ZEN ) ![](https://img.shields.io/github/stars/sinovation/ZEN.svg?style=social )|-|
|2019/11|[Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)|99|[Pytorch](https://github.com/facebookresearch/xlm ) ![](https://img.shields.io/github/stars/facebookresearch/xlm.svg?style=social )|XLM-R (XLM-RoBERTa)([xlmr.large](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz), [xlmr.base](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz))|
|2020/01|[ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063)|9|[Pytorch](https://github.com/microsoft/ProphetNet ) ![](https://img.shields.io/github/stars/microsoft/ProphetNet.svg?style=social )|ProphetNet([ProphetNet-large-16GB](https://drive.google.com/file/d/1PctDAca8517_weYUUBW96OjIPdolbQkd/view?usp=sharing), [ProphetNet-large-160GB](https://drive.google.com/file/d/1_nZcF-bBCQvBBcoPzA1nPZsz-Wo7hzEL/view?usp=sharing))|
|2020/02|[CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)|10|[Pytorch](https://github.com/microsoft/CodeBERT ) ![](https://img.shields.io/github/stars/microsoft/CodeBERT.svg?style=social )|[CodeBERT](https://drive.google.com/drive/folders/1MfkEkPlo_Cb8vZruOjbepNHEQHQEgoRm?usp=sharing )|
|2020/02|[UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training](https://arxiv.org/abs/2002.12804)|11|[Pytorch](https://github.com/microsoft/unilm/tree/master/unilm ) ![](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social )|-|
|2020/03|[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)|47|[TF](https://github.com/google-research/electra ) ![](https://img.shields.io/github/stars/google-research/electra.svg?style=social )|ELECTRA([ELECTRA-Small](https://storage.googleapis.com/electra-data/electra_small.zip), [ELECTRA-Base](https://storage.googleapis.com/electra-data/electra_base.zip), [ELECTRA-Large](https://storage.googleapis.com/electra-data/electra_large.zip))|
|2020/04|[MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)|2|[Pytorch](https://github.com/microsoft/MPNet ) ![](https://img.shields.io/github/stars/microsoft/MPNet.svg?style=social )|[MPNet](https://modelrelease.blob.core.windows.net/pre-training/MPNet/mpnet.base.tar.gz )|
|2020/05|[ParsBERT: Transformer-based Model for Persian Language Understanding](https://arxiv.org/abs/2005.12515)|0|[Pytorch](https://github.com/hooshvare/parsbert ) ![](https://img.shields.io/github/stars/hooshvare/parsbert.svg?style=social )|[ParsBERT](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased )|
|2020/05|[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)|56|-|-|
|2020/07|[InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training](https://arxiv.org/abs/2007.07834)|0|[Pytorch](https://github.com/microsoft/unilm/tree/master/infoxlm ) ![](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social )|-|

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
|-|[Incremental Domain Adaptation for Neural Machine Translation in Low-Resource Settings](https://www.aclweb.org/anthology/W19-4601)|N/A|[Python](https://github.com/DFKI-Interactive-Machine-Learning/AraSIF ) ![](https://img.shields.io/github/stars/DFKI-Interactive-Machine-Learning/AraSIF.svg?style=social )|AraSIF|
|2014/05|[Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053)|999+|[Pytorch](https://github.com/inejc/paragraph-vectors ) ![](https://img.shields.io/github/stars/inejc/paragraph-vectors.svg?style=social )<br>[Python](https://github.com/jhlau/doc2vec ) ![](https://img.shields.io/github/stars/jhlau/doc2vec.svg?style=social )|Doc2Vec|
|2014/11|[Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](https://arxiv.org/abs/1411.2539)|797|[Theano](https://github.com/ryankiros/visual-semantic-embedding ) ![](https://img.shields.io/github/stars/ryankiros/visual-semantic-embedding.svg?style=social )<br>[Pytorch](https://github.com/linxd5/VSE_Pytorch ) ![](https://img.shields.io/github/stars/linxd5/VSE_Pytorch.svg?style=social )|VSE|
|2015/06|[Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](https://arxiv.org/abs/1506.06724)|631|[Theano](https://github.com/ryankiros/skip-thoughts ) ![](https://img.shields.io/github/stars/ryankiros/skip-thoughts.svg?style=social )<br>[TF](https://github.com/tensorflow/models/tree/master/research/skip_thoughts ) ![](https://img.shields.io/github/stars/tensorflow/models.svg?style=social )<br>[Pytorch, Torch](https://github.com/Cadene/skip-thoughts.torch ) ![](https://img.shields.io/github/stars/Cadene/skip-thoughts.torch.svg?style=social )|SkipThought|
|2015/11|[Order-Embeddings of Images and Language](https://arxiv.org/abs/1511.06361)|322|[Theano](https://github.com/ivendrov/order-embedding ) ![](https://img.shields.io/github/stars/ivendrov/order-embedding.svg?style=social )|order-embedding|
|2015/11|[Towards Universal Paraphrastic Sentence Embeddings](https://arxiv.org/abs/1511.08198)|373|[Theano](https://github.com/jwieting/iclr2016 ) ![](https://img.shields.io/github/stars/jwieting/iclr2016.svg?style=social )|ParagramPhrase|
|2015/??|[From Word Embeddings to Document Distances](http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf)|941|[C, Python](https://github.com/mkusner/wmd ) ![](https://img.shields.io/github/stars/mkusner/wmd.svg?style=social )|Word Mover's Distance|
|2016/02|[Learning Distributed Representations of Sentences from Unlabelled Data](https://arxiv.org/abs/1602.03483)|334|[Python](https://github.com/fh295/SentenceRepresentation ) ![](https://img.shields.io/github/stars/fh295/SentenceRepresentation.svg?style=social )|FastSent|
|2016/07|[Charagram: Embedding Words and Sentences via Character n-grams](https://arxiv.org/abs/1607.02789)|137|[Theano](https://github.com/jwieting/charagram ) ![](https://img.shields.io/github/stars/jwieting/charagram.svg?style=social )|Charagram|
|2016/11|[Learning Generic Sentence Representations Using Convolutional Neural Networks](https://arxiv.org/abs/1611.07897)|70|[Theano](https://github.com/zhegan27/ConvSent ) ![](https://img.shields.io/github/stars/zhegan27/ConvSent.svg?style=social )|ConvSent|
|2017/03|[Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features](https://arxiv.org/abs/1703.02507)|272|[C++](https://github.com/epfml/sent2vec ) ![](https://img.shields.io/github/stars/epfml/sent2vec.svg?style=social )|Sent2Vec|
|2017/04|[Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444)|261|[TF](https://github.com/openai/generating-reviews-discovering-sentiment ) ![](https://img.shields.io/github/stars/openai/generating-reviews-discovering-sentiment.svg?style=social )<br>[Pytorch](https://github.com/guillitte/pytorch-sentiment-neuron ) ![](https://img.shields.io/github/stars/guillitte/pytorch-sentiment-neuron.svg?style=social )<br>[Pytorch](https://github.com/NVIDIA/sentiment-discovery ) ![](https://img.shields.io/github/stars/NVIDIA/sentiment-discovery.svg?style=social )|Sentiment Neuron|
|2017/05|[Revisiting Recurrent Networks for Paraphrastic Sentence Embeddings](https://arxiv.org/abs/1705.00364)|56|[Theano](https://github.com/jwieting/acl2017 ) ![](https://img.shields.io/github/stars/jwieting/acl2017.svg?style=social )|GRAN|
|2017/05|[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364)|907|[Pytorch](https://github.com/facebookresearch/InferSent ) ![](https://img.shields.io/github/stars/facebookresearch/InferSent.svg?style=social )|InferSent|
|2017/07|[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612)|119|[Pytorch](https://github.com/fartashf/vsepp ) ![](https://img.shields.io/github/stars/fartashf/vsepp.svg?style=social )|VSE++|
|2017/08|[Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](https://arxiv.org/abs/1708.00524)|308|[Keras](https://github.com/bfelbo/DeepMoji ) ![](https://img.shields.io/github/stars/bfelbo/DeepMoji.svg?style=social )<br>[Pytorch](https://github.com/huggingface/torchMoji ) ![](https://img.shields.io/github/stars/huggingface/torchMoji.svg?style=social )|DeepMoji|
|2017/09|[StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856)|111|[C++](https://github.com/facebookresearch/StarSpace ) ![](https://img.shields.io/github/stars/facebookresearch/StarSpace.svg?style=social )|StarSpace|
|2017/10|[DisSent: Learning Sentence Representations from Explicit Discourse Relations](https://arxiv.org/abs/1710.04334)|45|[Pytorch](https://github.com/windweller/DisExtract ) ![](https://img.shields.io/github/stars/windweller/DisExtract.svg?style=social )|DisSent|
|2017/11|[Dual-Path Convolutional Image-Text Embedding with Instance Loss](https://arxiv.org/abs/1711.05535)|52|[Matlab](https://github.com/layumi/Image-Text-Embedding ) ![](https://img.shields.io/github/stars/layumi/Image-Text-Embedding.svg?style=social )|Image-Text-Embedding|
|2017/11|[Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations](https://arxiv.org/abs/1711.05732)|98|[Theano](https://github.com/jwieting/para-nmt-50m ) ![](https://img.shields.io/github/stars/jwieting/para-nmt-50m.svg?style=social )|para-nmt|
|2018/03|[An efficient framework for learning sentence representations](https://arxiv.org/abs/1803.02893)|147|[TF](https://github.com/lajanugen/S2V ) ![](https://img.shields.io/github/stars/lajanugen/S2V.svg?style=social )|Quick-Thought|
|2018/03|[Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)|424|[TF-Hub](https://tfhub.dev/google/universal-sentence-encoder-large/2 )|USE|
|2018/04|[End-Task Oriented Textual Entailment via Deep Explorations of Inter-Sentence Interactions](https://arxiv.org/abs/1804.08813)|14|[Theano](https://github.com/yinwenpeng/SciTail ) ![](https://img.shields.io/github/stars/yinwenpeng/SciTail.svg?style=social )|DEISTE|
|2018/04|[Learning general purpose distributed sentence representations via large scale multi-task learning](https://arxiv.org/abs/1804.00079)|168|[Pytorch](https://github.com/Maluuba/gensen ) ![](https://img.shields.io/github/stars/Maluuba/gensen.svg?style=social )|GenSen|
|2018/06|[Embedding Text in Hyperbolic Spaces](https://arxiv.org/abs/1806.04313)|41|[TF](https://github.com/brain-research/hyperbolictext ) ![](https://img.shields.io/github/stars/brain-research/hyperbolictext.svg?style=social )|HyperText|
|2018/07|[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)|430|[Keras](https://github.com/davidtellez/contrastive-predictive-coding ) ![](https://img.shields.io/github/stars/davidtellez/contrastive-predictive-coding.svg?style=social )|CPC|
|2018/08|[Context Mover’s Distance & Barycenters: Optimal transport of contexts for building representations](https://arxiv.org/abs/1808.09663)|8|[Python](https://github.com/context-mover/context-mover-distance-and-barycenters ) ![](https://img.shields.io/github/stars/context-mover/context-mover-distance-and-barycenters.svg?style=social )|CMD|
|2018/09|[Learning Universal Sentence Representations with Mean-Max Attention Autoencoder](https://arxiv.org/abs/1809.06590)|9|[TF](https://github.com/Zminghua/SentEncoding ) ![](https://img.shields.io/github/stars/Zminghua/SentEncoding.svg?style=social )|Mean-MaxAAE|
|2018/10|[Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model](https://arxiv.org/abs/1810.12836)|23|[TF-Hub](https://tfhub.dev/s?q=universal-sentence-encoder-xling )|USE-xling|
|2018/10|[Improving Sentence Representations with Consensus Maximisation](https://arxiv.org/abs/1810.01064)|4|-|Multi-view|
|2018/10|[BioSentVec: creating sentence embeddings for biomedical texts](https://arxiv.org/abs/1810.09302)|47|[Python](https://github.com/ncbi-nlp/BioSentVec ) ![](https://img.shields.io/github/stars/ncbi-nlp/BioSentVec.svg?style=social )|BioSentVec|
|2018/11|[Word Mover's Embedding: From Word2Vec to Document Embedding](https://arxiv.org/abs/1811.01713)|38|[C, Python](https://github.com/IBM/WordMoversEmbeddings ) ![](https://img.shields.io/github/stars/IBM/WordMoversEmbeddings.svg?style=social )|WordMoversEmbeddings|
|2018/11|[A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks](https://arxiv.org/abs/1811.06031)|55|[Pytorch](https://github.com/huggingface/hmtl ) ![](https://img.shields.io/github/stars/huggingface/hmtl.svg?style=social )|HMTL|
|2018/12|[Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464)|159|[Pytorch](https://github.com/facebookresearch/LASER ) ![](https://img.shields.io/github/stars/facebookresearch/LASER.svg?style=social )|LASER|
|2018/??|[Convolutional Neural Network for Universal Sentence Embeddings](https://pdfs.semanticscholar.org/d827/32de6336dd6443ff33cccbb92ced0196ecc1.pdf)|5|[Theano](https://github.com/XiaoqiJiao/COLING2018 ) ![](https://img.shields.io/github/stars/XiaoqiJiao/COLING2018.svg?style=social )|CSE|
|2019/01|[No Training Required: Exploring Random Encoders for Sentence Classification](https://arxiv.org/abs/1901.10444)|47|[Pytorch](https://github.com/facebookresearch/randsent ) ![](https://img.shields.io/github/stars/facebookresearch/randsent.svg?style=social )|randsent|
|2019/02|[CBOW Is Not All You Need: Combining CBOW with the Compositional Matrix Space Model](https://arxiv.org/abs/1902.06423)|2|[Pytorch](https://github.com/florianmai/word2mat ) ![](https://img.shields.io/github/stars/florianmai/word2mat.svg?style=social )|CMOW|
|2019/07|[GLOSS: Generative Latent Optimization of Sentence Representations](https://arxiv.org/abs/1907.06385)|0|-|GLOSS|
|2019/07|[Multilingual Universal Sentence Encoder](https://arxiv.org/abs/1907.04307)|28|[TF-Hub](https://tfhub.dev/google/universal-sentence-encoder-multilingual/1 )|MultilingualUSE|
|2019/08|[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)|94|[Pytorch](https://github.com/UKPLab/sentence-transformers ) ![](https://img.shields.io/github/stars/UKPLab/sentence-transformers.svg?style=social )|Sentence-BERT|
|2020/06|[DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations](https://arxiv.org/abs/2006.03659)|0|[Pytorch](https://github.com/JohnGiorgi/DeCLUTR ) ![](https://img.shields.io/github/stars/JohnGiorgi/DeCLUTR.svg?style=social )|DeCLUTR|

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
