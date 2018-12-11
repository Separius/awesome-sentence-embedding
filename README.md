# awesome-sentence-embedding
A curated list of awesome sentence embedding models

## Table of Contents

* **[About This Repo](#about-this-repo)**
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

## About This Repo
* well there are some awesome-lists for word embeddings and sentence embeddings, but all of them are outdated and more importantly incomplete
* this repo will also be incomplete! but I try my best to find and include all the papers with pretrained models(so I have a strong bias toward papers with code and pretrained models)
* this is not a typical awesome list, it has tables! but I guess it's ok and much better than just some lists
* if you find any mistakes or find another paper or anything please send a pull request and help me to keep this list up to date
* to be honest I'm not 100% sure how to represent this data and if you think there is a better way (for example by changing the table headers) please send a pull request and let us discuss it
* enjoy!

## General Framework

* Almost all the sentence embeddings work like this: given some sort of word embeddings and an optional encoder (for example an LSTM) they obtain the contextualized word embeddings and then they define some sort of pooling (it can be as simple as last pooling) and then based on that they either use it directly for the supervised classification task (like infersent) or generate the target sequence (like skip-thought) so in general we have many sentence embeddings that you have never heard of, you can simply do mean-pooling over any word embedding and it's a sentence embedding!

## Word Embeddings

* Note: don't worry about the language of the code, you can almost always (except for the subword models) just use the pretrained embedding table in the framework of your choice and ignore the training code

|paper|training code|pretrained models|
|---|---|---|
|[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)|[C++](https://github.com/stanfordnlp/GloVe)(official)|[GloVe](https://nlp.stanford.edu/projects/glove/)|
|[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)|[C++](https://code.google.com/archive/p/word2vec/)(official)|[Word2Vec](https://code.google.com/archive/p/word2vec/)|
|[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)|[C++](https://github.com/facebookresearch/fastText)(official)|[fastText](https://fasttext.cc/docs/en/english-vectors.html)|
|[BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages](https://arxiv.org/pdf/1710.02187.pdf)|[Python](https://github.com/bheinzerling/bpemb)(official)|[bpemb](https://github.com/bheinzerling/bpemb#downloads-for-each-language)|
|[ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://arxiv.org/pdf/1612.03975.pdf)|[Python](https://github.com/commonsense/conceptnet-numberbatch)(official)|[Numberbatch](https://github.com/commonsense/conceptnet-numberbatch#downloads)|
|[A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks](https://arxiv.org/pdf/1611.01587.pdf)|<ul><li>[C++](https://github.com/hassyGo/charNgram2vec)(Official)</li><li>[Pytorch](https://github.com/hassyGo/pytorch-playground/tree/master/jmt)</li></ul>|[charNgram2vec](http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz)|
|[Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations](http://anthology.aclweb.org/P16-2068)|[GO](https://github.com/alexandres/lexvec)(official)|[lexvec](https://github.com/alexandres/lexvec#pre-trained-vectors)|
|[Hash Embeddings for Efficient Word Representations](https://arxiv.org/pdf/1709.03933.pdf)|<ul><li>[Keras](https://github.com/dsv77/hashembedding)(official)</li><li>[Pytorch](https://github.com/YannDubs/Hash-Embeddings)</li></ul>|-|
|[Dependency-Based Word Embeddings](http://www.aclweb.org/anthology/P14-2050)|<ul><li>[C++](https://bitbucket.org/yoavgo/word2vecf/src/default/)(official)</li><li>[DL4J](https://github.com/IsaacChanghau/Word2VecfJava)</li></ul>|[word2vecf](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)|
|[Learning Word Meta-Embeddings](http://www.aclweb.org/anthology/P16-1128)|-|[Meta-Emb](http://cistern.cis.lmu.de/meta-emb/)(broken)|
|[Dict2vec : Learning Word Embeddings using Lexical Dictionaries](http://aclweb.org/anthology/D17-1024)|[C++](https://github.com/tca19/dict2vec)(official)|[Dict2vec](https://github.com/tca19/dict2vec#download-pre-trained-vectors)|
|[Semantic Specialisation of Distributional Word Vector Spaces using Monolingual and Cross-Lingual Constraints](https://arxiv.org/pdf/1706.00374)|[TF](https://github.com/nmrksic/attract-repel)(official)|[Attract-Repel](https://github.com/nmrksic/attract-repel#available-word-vector-spaces)(bilingual)|
|[Siamese CBOW: Optimizing Word Embeddings for Sentence Representations](https://arxiv.org/pdf/1606.04640)|<ul><li>[Theano](https://bitbucket.org/TomKenter/siamese-cbow/src/master/)(official)</li><li>[TF](https://github.com/raphael-sch/SiameseCBOW)</li></ul>|[Siamese CBOW](https://bitbucket.org/TomKenter/siamese-cbow/src/master/)|
|[Offline bilingual word vectors, orthogonal transformations and the inverted softmax](https://arxiv.org/pdf/1702.03859)|[Python](https://github.com/Babylonpartners/fastText_multilingual)(official)|-|
|[From Paraphrase Database to Compositional Paraphrase Model and Back](http://www.aclweb.org/anthology/Q15-1025)|[Theano](https://github.com/jwieting/paragram-word)(official)|[PARAGRAM](http://ttic.uchicago.edu/~wieting/paragram-word-demo.zip)|

## OOV Handling

* Drop OOV words!
* One OOV vector(unk vector)
* [ALaCarte](https://github.com/NLPrinceton/ALaCarte): [A La Carte Embedding: Cheap but Effective Induction of Semantic Feature Vectors](http://aclweb.org/anthology/P18-1002)
* [Mimick](https://github.com/yuvalpinter/Mimick): [Mimicking Word Embeddings using Subword RNNs](http://www.aclweb.org/anthology/D17-1010)

## Contextualized Word Embeddings

* Note: all the unofficial models can load the official pretrained models

|paper|code|pretrained models|
|---|---|---|
|[Learned in Translation: Contextualized Word Vectors](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf)|<ul><li>[Pytorch](https://github.com/salesforce/cove)(official)</li><li>[Keras](https://github.com/rgsachin/CoVe)</li></ul>|[CoVe](https://github.com/salesforce/cove)|
|[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365)|<ul><li>[Pytorch](https://github.com/allenai/allennlp)(official)</li><li>[TF](https://github.com/allenai/bilm-tf)(official)</li>|ELMO([AllenNLP](https://allennlp.org/elmo), [TF-Hub](https://tfhub.dev/google/elmo/2))|
|[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)|<ul><li>[TF](https://github.com/openai/finetune-transformer-lm)(official)</li><li>[Keras](https://github.com/Separius/BERT-keras)</li><li>[Pytorch](https://github.com/huggingface/pytorch-openai-transformer-lm)</li></ul>|[Transformer](https://github.com/openai/finetune-transformer-lm)
|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)|<ul><li>[TF](https://github.com/google-research/bert)(official)</li><li>[Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT)</li><li>[Keras](https://github.com/Separius/BERT-keras)</li></ul>|[BERT](https://github.com/google-research/bert#pre-trained-models)|

## Pooling Methods
* {Last, Mean, Max}-Pooling
* Special Token Pooling (like BERT and OpenAI's Transformer)
* [SIF](https://github.com/PrincetonML/SIF): [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx)
* [TF-IDF](https://github.com/iarroyof/sentence_embedding): [Unsupervised Sentence Representations as Word Information Series: Revisiting TF--IDF](https://arxiv.org/pdf/1710.06524)
* [P-norm](https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings): [Concatenated Power Mean Word Embeddings as Universal Cross-Lingual Sentence Representations](https://arxiv.org/pdf/1803.01400)

## Encoders
|paper|code|name|
|---|---|---|
|[An efficient framework for learning sentence representations](https://arxiv.org/pdf/1803.02893.pdf)|[TF](https://github.com/lajanugen/S2V)(official, pretrained)|Quick-Thought|
|[Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](https://arxiv.org/pdf/1708.00524)|<ul><li>[Keras](https://github.com/bfelbo/DeepMoji)(official, pretrained)</li><li>[Pytorch](https://github.com/huggingface/torchMoji)(load_pretrained)</li></ul>|DeepMoji|
|[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/pdf/1705.02364)|[Pytorch](https://github.com/facebookresearch/InferSent)(official, pretrained)|InferSent|
|[Learning Joint Multilingual Sentence Representations with Neural Machine Translation](https://aclanthology.info/papers/W17-2619/w17-2619)|[Pytorch](https://github.com/facebookresearch/LASER)(official, pretrained)|LASER|
|[Learning general purpose distributed sentence representations via large scale multi-task learning](https://arxiv.org/pdf/1804.00079)|[Pytorch](https://github.com/Maluuba/gensen)(official, pretrained)|GenSen|
|[Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053)|<ul><li>[Pytorch](https://github.com/inejc/paragraph-vectors)</li><li>[Python](https://github.com/jhlau/doc2vec)(pretrained)</li></ul>|Doc2Vec|
|[Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features](https://arxiv.org/pdf/1703.02507.pdf)|[C++](https://github.com/epfml/sent2vec)(official, pretrained)|Sent2Vec|
|[Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](https://arxiv.org/pdf/1506.06724)|<ul><li>[Theano](https://github.com/ryankiros/skip-thoughts)(official, pretrained)</li><li>[TF](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)(pretrained)</li><li>[Pytorch,Torch](https://github.com/Cadene/skip-thoughts.torch)(load_pretrained)</li></ul>|SkipThought|
|[Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/pdf/1704.01444)|<ul><li>[TF](https://github.com/openai/generating-reviews-discovering-sentiment)(official, pretrained)</li><li>[Pytorch](https://github.com/guillitte/pytorch-sentiment-neuron)(load_pretrained)</li><li>[Pytorch](https://github.com/NVIDIA/sentiment-discovery)(pretrained)</li></ul>|SentimentNeuron|
|[From Word Embeddings to Document Distances](http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf)|[C,Python](https://github.com/mkusner/wmd)(official)|Word Mover's Distance|
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
|[Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)|[TF-Hub](https://tfhub.dev/google/universal-sentence-encoder/2)(official, pretrained)|USE|
