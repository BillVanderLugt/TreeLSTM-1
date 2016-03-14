# TreeLSTM
An attempt to implement the Constinuency Tree LSTM in "[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075)" (Tai, Socher, and Manning, 2015) in Theano.

Python requirements:
===
- Python 2.7
- NumPy
- Theano
- gensim

Data requirements:
===
- Stanford Sentiment Treebank in PTB Tree format: [download](http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip)
- pretrained word2vec word vectors trained on Google News, available [here](https://code.google.com/archive/p/word2vec/)

Installation:
===
- download the source files or clone the repo
- download the Stanford Sentiment Treebank in PTB format using the above link
- download the pre-trained word2vec vectors using the above link, and unzip the file

To run:
===
To run using default settings:

```
python constituency_tree_lstm.py /path/to/stanfordSentimentTreebank/trees --word2vec_file=/path/to/GoogleNews-vectors-negative300.bin
```

To see additional options:

```
python constituency_tree_lstm.py -h
```

Differences from Original Implementation
===
The original Constituency Tree LSTM was implemented in Torch. In addition, the following are different about this implementation

- no minibatches
- adjusted learning rate and regularization strength to compensate for minibatches of size=1
- using word2vec vectors, instead of Glove
- no updatting of embeddings during training

Performance
===
Performance is not yet equivalent to the results of the original paper, perhaps because of the above differences. The best performance I have been able to obtain thus far is ~0.85 on the binary classification task.

