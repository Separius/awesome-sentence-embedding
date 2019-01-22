# awesome-sentence-embedding

A curated list of pretrained sentence(and word) embedding models

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
* **[Code Less](#code-less)**

## About This Repo

* well there are some awesome-lists for word embeddings and sentence embeddings, but all of them are outdated and more importantly incomplete
* this repo will also be incomplete, but I try my best to find and include all the papers with pretrained models
* this is not a typical awesome list because it has tables but I guess it's ok and much better than just a huge list
* if you find any mistakes or find another paper or anything please send a pull request and help me to keep this list up to date
* to be honest I'm not 100% sure how to represent this data and if you think there is a better way (for example by changing the table headers) please send a pull request and let us discuss it
* enjoy!

## General Framework

* Almost all the sentence embeddings work like this: 
* Given some sort of word embeddings and an optional encoder (for example an LSTM) they obtain the contextualized word embeddings. 
* Then they define some sort of pooling (it can be as simple as last pooling). 
* Based on that they either use it directly for the supervised classification task (like infersent) or generate the target sequence (like skip-thought). 
* So, in general, we have many sentence embeddings that you have never heard of, you can simply do mean-pooling over any word embedding and it's a sentence embedding!

## Word Embeddings

* Note: don't worry about the language of the code, you can almost always (except for the subword models) just use the pretrained embedding table in the framework of your choice and ignore the training code

|paper|training code|pretrained models|
|---|---|---|
|[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)|[C](https://github.com/stanfordnlp/GloVe)(official)|[GloVe](https://nlp.stanford.edu/projects/glove/)|
|[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)|[C](https://github.com/tmikolov/word2vec)(official)|[Word2Vec](https://code.google.com/archive/p/word2vec/)|
|[Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)|[C++](https://github.com/facebookresearch/fastText)(official)|[fastText](https://fasttext.cc/docs/en/english-vectors.html)|
|[BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages](https://arxiv.org/pdf/1710.02187.pdf)|[Python](https://github.com/bheinzerling/bpemb)(official)|[bpemb](https://github.com/bheinzerling/bpemb#downloads-for-each-language)|
|[ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://arxiv.org/pdf/1612.03975.pdf)|[Python](https://github.com/commonsense/conceptnet-numberbatch)(official)|[Numberbatch](https://github.com/commonsense/conceptnet-numberbatch#downloads)|
|[Non-distributional Word Vector Representations](https://arxiv.org/pdf/1506.05230.pdf)|[Python](https://github.com/mfaruqui/non-distributional)(official)|[WordFeat](https://github.com/mfaruqui/non-distributional)|
|[Sparse Overcomplete Word Vector Representations](https://arxiv.org/pdf/1506.02004)|[C++](https://github.com/mfaruqui/sparse-coding)(official)|-|
|[A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks](https://arxiv.org/pdf/1611.01587.pdf)|<ul><li>[C++](https://github.com/hassyGo/charNgram2vec)(official)</li><li>[Pytorch](https://github.com/hassyGo/pytorch-playground/tree/master/jmt)</li></ul>|[charNgram2vec](http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz)|
|[Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations](http://anthology.aclweb.org/P16-2068)|[GO](https://github.com/alexandres/lexvec)(official)|[lexvec](https://github.com/alexandres/lexvec#pre-trained-vectors)|
|[Hash Embeddings for Efficient Word Representations](https://arxiv.org/pdf/1709.03933.pdf)|<ul><li>[Keras](https://github.com/dsv77/hashembedding)(official)</li><li>[Pytorch](https://github.com/YannDubs/Hash-Embeddings)</li></ul>|-|
|[Dependency-Based Word Embeddings](http://www.aclweb.org/anthology/P14-2050)|<ul><li>[C++](https://bitbucket.org/yoavgo/word2vecf/src/default/)(official)</li><li>[DL4J](https://github.com/IsaacChanghau/Word2VecfJava)</li></ul>|[word2vecf](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)|
|[Learning Word Meta-Embeddings](http://www.aclweb.org/anthology/P16-1128)|-|[Meta-Emb](http://cistern.cis.lmu.de/meta-emb/)(broken)|
|[Dict2vec : Learning Word Embeddings using Lexical Dictionaries](http://aclweb.org/anthology/D17-1024)|[C++](https://github.com/tca19/dict2vec)(official)|[Dict2vec](https://github.com/tca19/dict2vec#download-pre-trained-vectors)|
|[Semantic Specialisation of Distributional Word Vector Spaces using Monolingual and Cross-Lingual Constraints](https://arxiv.org/pdf/1706.00374)|[TF](https://github.com/nmrksic/attract-repel)(official)|[Attract-Repel](https://github.com/nmrksic/attract-repel#available-word-vector-spaces)|
|[Siamese CBOW: Optimizing Word Embeddings for Sentence Representations](https://arxiv.org/pdf/1606.04640)|<ul><li>[Theano](https://bitbucket.org/TomKenter/siamese-cbow/src/master/)(official)</li><li>[TF](https://github.com/raphael-sch/SiameseCBOW)</li></ul>|[Siamese CBOW](https://bitbucket.org/TomKenter/siamese-cbow/src/master/)|
|[Offline bilingual word vectors, orthogonal transformations and the inverted softmax](https://arxiv.org/pdf/1702.03859)|[Python](https://github.com/Babylonpartners/fastText_multilingual)(official)|-|
|[From Paraphrase Database to Compositional Paraphrase Model and Back](http://www.aclweb.org/anthology/Q15-1025)|[Theano](https://github.com/jwieting/paragram-word)(official)|[PARAGRAM](http://ttic.uchicago.edu/~wieting/paragram-word-demo.zip)|
|[Poincaré Embeddings for Learning Hierarchical Representations](https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations.pdf)|[Pytorch](https://github.com/facebookresearch/poincare-embeddings)(official)|-|
|[Dynamic Meta-Embeddings for Improved Sentence Representations](https://arxiv.org/pdf/1804.07983.pdf)|[Pytorch](https://github.com/facebookresearch/DME)(official)|[DME/CDME](https://github.com/facebookresearch/DME#pre-trained-models)|
|[WebVectors: A Toolkit for Building Web Interfaces for Vector Semantic Models](https://rusvectores.org/static/data/webvectors_aist.pdf)|-|[RusVectōrēs](http://rusvectores.org/en/models/)|
|[Swivel: Improving Embeddings by Noticing What's Missing](https://arxiv.org/pdf/1602.02215)|[TF](https://github.com/tensorflow/models/tree/master/research/swivel)(official)|-|
|[Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings](https://ai.tencent.com/ailab/media/publications/naacl2018/directional_skip-gram.pdf)|-|[ChineseEmbedding](https://ai.tencent.com/ailab/nlp/embedding.html)|

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
|[Learned in Translation: Contextualized Word Vectors](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf)|<ul><li>[Pytorch](https://github.com/salesforce/cove)(official)</li><li>[Keras](https://github.com/rgsachin/CoVe)</li></ul>|[CoVe](https://github.com/salesforce/cove)|
|[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)|[Pytorch](https://github.com/fastai/fastai/tree/ulmfit_v1)(official)|ULMFit([English](https://docs.fast.ai/text.html#Fine-tuning-a-language-model), [Zoo](https://forums.fast.ai/t/language-model-zoo-gorilla/14623/1))|
|[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365)|<ul><li>[Pytorch](https://github.com/allenai/allennlp)(official)</li><li>[TF](https://github.com/allenai/bilm-tf)(official)</li>|ELMO([AllenNLP](https://allennlp.org/elmo), [TF-Hub](https://tfhub.dev/google/elmo/2))|
|[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)|<ul><li>[TF](https://github.com/openai/finetune-transformer-lm)(official)</li><li>[Keras](https://github.com/Separius/BERT-keras)</li><li>[Pytorch](https://github.com/huggingface/pytorch-openai-transformer-lm)</li></ul>|[Transformer](https://github.com/openai/finetune-transformer-lm)
|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)|<ul><li>[TF](https://github.com/google-research/bert)(official)</li><li>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT)</li><li>[Keras](https://github.com/Separius/BERT-keras)</li></ul>|[BERT](https://github.com/google-research/bert#pre-trained-models)|
|[Towards Better UD Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation](https://arxiv.org/pdf/1807.03121)|[Pytorch](https://github.com/HIT-SCIR/ELMoForManyLangs)(official)|[ELMo](https://github.com/HIT-SCIR/ELMoForManyLangs#downloads)|
|[Contextual String Embeddings for Sequence Labeling](http://aclweb.org/anthology/C18-1139)|[Pytorch](https://github.com/zalandoresearch/flair)(official)|[Flair](https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py#L407)|
|[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860)|<ul><li>[TF](https://github.com/kimiyoung/transformer-xl/tree/master/tf)(official)</li><li>[Pytorch](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch)(official)</li></ul>|[Transformer-XL](https://github.com/kimiyoung/transformer-xl/tree/master/tf)|

## Pooling Methods

* {Last, Mean, Max}-Pooling
* Special Token Pooling (like BERT and OpenAI's Transformer)
* [SIF](https://github.com/PrincetonML/SIF): [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx)
* [TF-IDF](https://github.com/iarroyof/sentence_embedding): [Unsupervised Sentence Representations as Word Information Series: Revisiting TF--IDF](https://arxiv.org/pdf/1710.06524)
* [P-norm](https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings): [Concatenated Power Mean Word Embeddings as Universal Cross-Lingual Sentence Representations](https://arxiv.org/pdf/1803.01400)
* [DisC](https://github.com/NLPrinceton/text_embedding): [A Compressed Sensing View of Unsupervised Text Embeddings, Bag-of-n-Grams, and LSTMs](https://openreview.net/pdf?id=B1e5ef-C-)

## Encoders

|paper|code|name|
|---|---|---|
|[An efficient framework for learning sentence representations](https://arxiv.org/pdf/1803.02893.pdf)|[TF](https://github.com/lajanugen/S2V)(official, pretrained)|Quick-Thought|
|[Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](https://arxiv.org/pdf/1708.00524)|<ul><li>[Keras](https://github.com/bfelbo/DeepMoji)(official, pretrained)</li><li>[Pytorch](https://github.com/huggingface/torchMoji)(load_pretrained)</li></ul>|DeepMoji|
|[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/pdf/1705.02364)|[Pytorch](https://github.com/facebookresearch/InferSent)(official, pretrained)|InferSent|
|[Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/pdf/1812.10464.pdf)|[Pytorch](https://github.com/facebookresearch/LASER)(official, pretrained)|LASER|
|[Learning general purpose distributed sentence representations via large scale multi-task learning](https://arxiv.org/pdf/1804.00079)|[Pytorch](https://github.com/Maluuba/gensen)(official, pretrained)|GenSen|
|[Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053)|<ul><li>[Pytorch](https://github.com/inejc/paragraph-vectors)</li><li>[Python](https://github.com/jhlau/doc2vec)(pretrained)</li></ul>|Doc2Vec|
|[Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features](https://arxiv.org/pdf/1703.02507.pdf)|[C++](https://github.com/epfml/sent2vec)(official, pretrained)|Sent2Vec|
|[Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](https://arxiv.org/pdf/1506.06724)|<ul><li>[Theano](https://github.com/ryankiros/skip-thoughts)(official, pretrained)</li><li>[TF](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)(pretrained)</li><li>[Pytorch,Torch](https://github.com/Cadene/skip-thoughts.torch)(load_pretrained)</li></ul>|SkipThought|
|[Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/pdf/1704.01444)|<ul><li>[TF](https://github.com/openai/generating-reviews-discovering-sentiment)(official, pretrained)</li><li>[Pytorch](https://github.com/guillitte/pytorch-sentiment-neuron)(load_pretrained)</li><li>[Pytorch](https://github.com/NVIDIA/sentiment-discovery)(pretrained)</li></ul>|SentimentNeuron|
|[From Word Embeddings to Document Distances](http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf)|[C,Python](https://github.com/mkusner/wmd)(official)|Word Mover's Distance|
|[Word Mover's Embedding: From Word2Vec to Document Embedding](https://arxiv.org/pdf/1811.01713)|[C,Python](https://github.com/IBM/WordMoversEmbeddings)(official)|WordMoversEmbeddings|
|[Convolutional Neural Network for Universal Sentence Embeddings](https://pdfs.semanticscholar.org/d827/32de6336dd6443ff33cccbb92ced0196ecc1.pdf)|[Theano](https://github.com/XiaoqiJiao/COLING2018)(official, pretrained)|CSE|
|[Towards Universal Paraphrastic Sentence Embeddings](https://arxiv.org/pdf/1511.08198)|[Theano](https://github.com/jwieting/iclr2016)(official, pretrained)|ParagramPhrase|
|[Charagram: Embedding Words and Sentences via Character n-grams](https://aclweb.org/anthology/D16-1157)|[Theano](https://github.com/jwieting/charagram)(official, pretrained)|Charagram|
|[Revisiting Recurrent Networks for Paraphrastic Sentence Embeddings](https://arxiv.org/pdf/1705.00364)|[Theano](https://github.com/jwieting/acl2017)(official, pretrained)|GRAN|
|[Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations](https://arxiv.org/pdf/1711.05732)|[Theano](https://github.com/jwieting/para-nmt-50m)(official, pretrained)|para-nmt|
|[Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](https://arxiv.org/pdf/1411.2539)|<ul><li>[Theano](https://github.com/ryankiros/visual-semantic-embedding)(official, pretrained)</li><li>[Pytorch](https://github.com/linxd5/VSE_Pytorch)(load_pretrained)</li></ul>|VSE|
|[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/pdf/1707.05612)|[Pytorch](https://github.com/fartashf/vsepp)(official, pretrained)|VSE++|
|[End-Task Oriented Textual Entailment via Deep Explorations of Inter-Sentence Interactions](https://arxiv.org/pdf/1804.08813)|[Theano](https://github.com/yinwenpeng/SciTail)(official, pretrained)|DEISTE|
|[Learning Universal Sentence Representations with Mean-Max Attention Autoencoder](http://aclweb.org/anthology/D18-1481)|[TF](https://github.com/Zminghua/SentEncoding)(official, pretrained)|Mean-MaxAAE|
|[BioSentVec: creating sentence embeddings for biomedical texts](https://arxiv.org/pdf/1810.09302)|[Python](https://github.com/ncbi-nlp/BioSentVec)(official, pretrained)|BioSentVec|
|[DisSent: Learning Sentence Representations from Explicit Discourse Relations](https://arxiv.org/pdf/1710.04334.pdf)|[Pytorch](https://github.com/windweller/DisExtract)(official, email_for_pretrained)|DisSent|
|[Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf)|[TF-Hub](https://tfhub.dev/google/universal-sentence-encoder/2)(official, pretrained)|USE|
|[Learning Distributed Representations of Sentences from Unlabelled Data](https://arxiv.org/pdf/1602.03483)|[Python](https://github.com/fh295/SentenceRepresentation)(official)|FastSent|
|[Embedding Text in Hyperbolic Spaces](https://arxiv.org/pdf/1806.04313.pdf)|[TF](https://github.com/brain-research/hyperbolictext)(official)|HyperText|
|[StarSpace: Embed All The Things!](https://arxiv.org/pdf/1709.03856.pdf)|[C++](https://github.com/facebookresearch/StarSpace)(official)|StarSpace|
|[A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks](https://arxiv.org/pdf/1811.06031.pdf)|[Pytorch](https://github.com/huggingface/hmtl)(official, pretrained)|HMTL|

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

## Misc

* [Word Embedding Dimensionality Selection](https://github.com/ziyin-dl/word-embedding-dimensionality-selection): [On the Dimensionality of Word Embedding](https://arxiv.org/pdf/1812.04224.pdf)

## Vector Mapping

* [Cross-lingual Word Vectors Projection Using CCA](https://github.com/mfaruqui/crosslingual-cca): [Improving Vector Space Word Representations Using Multilingual Correlation](http://www.aclweb.org/anthology/E14-1049)
* [vecmap](https://github.com/artetxem/vecmap): [A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings](https://arxiv.org/pdf/1805.06297)
* [MUSE](https://github.com/facebookresearch/MUSE): [Unsupervised Machine Translation Using Monolingual Corpora Only](https://arxiv.org/pdf/1711.00043)

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

## Code Less

* papers here are just a paper and they don't have any released codes or pretrained models
* are you sure? I have read the paper, googled the title, googled the title + github, and searched for the authors one by one and checked their pages, so yeah I'm 60% sure that they don't have anything! :))
* I did these two months ago(Oct 2018), and they might have released their codes in this time, so If you found any of them let me know.
* [TOWARDS LANGUAGE AGNOSTIC UNIVERSAL REPRESENTATIONS](https://arxiv.org/pdf/1809.08510.pdf)
* [IS WASSERSTEIN ALL YOU NEED?](https://arxiv.org/pdf/1808.09663.pdf)
* [Discourse-Based Objectives for Fast Unsupervised Sentence Representation Learning](https://arxiv.org/pdf/1705.00557.pdf)
* [UNSUPERVISED SENTENCE EMBEDDING USING DOCUMENT STRUCTURE-BASED CONTEXT](https://openreview.net/pdf?id=H1a37GWCZ)
* [CSE: Conceptual Sentence Embeddings based on Attention Model](https://www.aclweb.org/anthology/P/P16/P16-1048.pdf)
* [Unsupervised Document Embedding With CNNs](https://arxiv.org/pdf/1711.04168.pdf)
* [Learning Generic Sentence Representations Using Convolutional Neural Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/sent2vec_emnlp2017.pdf)
* [Learning Cross-Lingual Sentence Representations via a Multi-task Dual-Encoder Model](https://arxiv.org/pdf/1810.12836.pdf)
* [ZERO-TRAINING SENTENCE EMBEDDING VIA ORTHOGONAL BASIS](https://arxiv.org/pdf/1810.00438.pdf)
* [Improving Sentence Representations with Multi-view Frameworks](https://arxiv.org/pdf/1810.01064.pdf)
* [UNSUPERVISED LEARNING OF SENTENCE REPRESENTATIONS USING SEQUENCE CONSISTENCY](https://openreview.net/pdf?id=BklUAoAcY7)
* [FAKE SENTENCE DETECTION AS A TRAINING TASK FOR SENTENCE ENCODING](https://openreview.net/pdf?id=B1xFVhActm)
* [POINCARE´ GLOVE: HYPERBOLIC WORD EMBEDDINGS](https://openreview.net/pdf?id=Ske5r3AqK7)
* [A NON-LINEAR THEORY FOR SENTENCE EMBEDDING](openreview.net/pdf?id=SJMZRsC9Y7)
* [NO TRAINING REQUIRED: EXPLORING RANDOM ENCODERS FOR SENTENCE CLASSIFICATION](openreview.net/pdf?id=BkgPajAcY7)
* [VARIATIONAL AUTOENCODERS FOR TEXT MODELING WITHOUT WEAKENING THE DECODER](openreview.net/pdf?id=H1eZ6sRcFm)
* [IMPROVING COMPOSITION OF SENTENCE EMBEDDINGS THROUGH THE LENS OF STATISTICAL RELATIONAL LEARNING](https://openreview.net/pdf?id=SkxZFoAqtQ)
* [InferLite : Simple Universal Sentence Representations from Natural Language Inference Data](http://aclweb.org/anthology/D18-1524)

