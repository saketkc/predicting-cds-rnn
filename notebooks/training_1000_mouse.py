
# coding: utf-8

# In[33]:


import importlib
import os
import json
import pickle
import sys
import numpy as np
import pandas as pd
import tensorflow as tf


from keras_tqdm import TQDMNotebookCallback
from keras.models import load_model
from keras.utils import to_categorical
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
#Dropout, RepeatVector, TimeDistributed, AveragePooling1D, Flatten
from collections import defaultdict, OrderedDict
from scipy.stats import describe
print(K.backend())
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend


#set_keras_backend('theano')
__BASES__ = ['A','C','G','T']
__BASES_MAP__ = OrderedDict(zip(__BASES__,range(4)))
__MERGE_KEYS__ = ['UTR5', 'CDS', 'UTR3']
__MERGE_LABELS__ = OrderedDict(zip(__MERGE_KEYS__,range(3)))


# In[21]:


def _downsample(genes_histogram, genes_to_keep=1000):
    """Downsample a histogram by randomly dropping proportional
    number of genes in each bin

    Params
    ------

    genes_histogram : dict
        Dictionary with format {bin:[list of genes in bin]}

    genes_to_keep : int
        Total genes to keep

    Return
    ------
    downsampled_dict : dict
        Dictionary with downsampled list in each bin

    """
    np.random.seed(42)
    downsampled_dict = {}
    total_bins = len(genes_histogram)
    total_genes = sum([len(x) for x in list(genes_histogram.values())])
    scaling_factor = genes_to_keep / total_genes
    for bin_index, genes_in_bin in list(genes_histogram.items()):
        n_genes_in_bin = len(genes_in_bin)
        n_genes_to_keep = int(np.ceil(n_genes_in_bin * scaling_factor))
        index_genes_to_keep = np.random.choice(n_genes_in_bin, n_genes_to_keep)
        genes_to_keep = np.array(genes_in_bin)[index_genes_to_keep]
        downsampled_dict[bin_index] = list(genes_to_keep)
    return downsampled_dict


def load_data(gene_cds, gene_lengths, genes_to_keep=1000):
    """Load dataset

    Params
    ------
    gene_cds : str
        Path to json with sequence

    gene_lengths : str
        Path to json with sequence lengths

    genes_to_keep : int
        Total genes_to_keep

    Return
    ------
    downsampled_genes_dict : dict
        Dictionary with format {bin:[list of genes in bin]}

    """
    np.random.seed(42)

    gene_seq = OrderedDict(json.load(open(gene_cds)))
    gene_len = OrderedDict(json.load(open(gene_lengths)))

    gene_total_len = OrderedDict((k, sum(list(v.values())))
                                 for k, v in list(gene_len.items()))
    all_lengths = np.array(list(gene_total_len.values()))
    valid_genes_dict = OrderedDict((k, v)
                                   for k, v in list(gene_total_len.items())
                                   if v < 10000)

    valid_genes_keys = list(valid_genes_dict.keys())
    valid_genes_values = list(valid_genes_dict.values())
    hist, edges = np.histogram(valid_genes_values)
    valid_genes_bins = np.digitize(valid_genes_values, edges)
    length_wise_binned_genes = defaultdict(list)
    for i, b in enumerate(valid_genes_bins):
        length_wise_binned_genes[b - 1].append(valid_genes_keys[i])

    downsampled_genes_dict = _downsample(length_wise_binned_genes,
                                         genes_to_keep=genes_to_keep)
    return downsampled_genes_dict, gene_seq


def split_train_test_genes(length_wise_binned_genes, train_proportion=0.7):
    """Split data in training and testing set

    Params
    ------
    length_wise_binned_genes : dict
         Dictionary with format {bin:[list of genes in bin]}

    train_proportion : float
        Training proportion

    Return
    ------
    training_genes : list

    testing_genes : list

    """
    np.random.seed(42)
    training_genes = []
    testing_genes = []
    for bin_number, bin_genes in list(length_wise_binned_genes.items()):
        n_genes = len(bin_genes)
        np.random.shuffle(bin_genes)
        training_genes += bin_genes[:int(n_genes * train_proportion)]
        testing_genes += bin_genes[int(n_genes * train_proportion):]

    return training_genes, testing_genes


# In[9]:


def map_base_to_int(base):
    """Return int given a base character"""
    return __BASES_MAP__[base]


def one_hot_encoding(data_dict):
    """One hot encode sequences

    Params
    ------
    data_dict : dict
        Sequence dict as loaded from gene_cds json file

    Returns
    -------

    X : array
        X*4 Input array with columns representing A,T,G,C

    Y : array
        X*3 Labels with columns representing 5'UTR, CDS, 3'UTR


    """
    merged_seq = []
    merged_label = []
    for key in __MERGE_KEYS__:
        merged_seq += list(data_dict[key])
        merged_label += list([__MERGE_LABELS__[key]] * len(data_dict[key]))
    merged_seq_int = list(map(map_base_to_int, merged_seq))

    X = to_categorical(merged_seq_int)
    Y = to_categorical(merged_label)
    return X, Y


# In[49]:


def train(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(
        LSTM(
            3,
            return_sequences=True,
            input_shape=(None, 4),
            dropout=0.25,
            recurrent_dropout=0.25))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
    nb_epoch = 128
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for e in range(nb_epoch):
        index = 0
        acc_train = 0
        for x, y in zip(X_train, Y_train):
            acc = model.train_on_batch(np.array([x]), np.array([y]))
            train_history[e].append(acc)
            acc_train+=acc[1]
            if (index%500) == 0:
                sys.stderr.write('Epoch: {} || Index :{} || loss: {} || acc: {}\n'.format(e, index, acc[0], acc[1]))
                with open('train_acc_1000_mouse.log', 'a') as f:
                    f.write('Epoch: {} || Index :{} || loss: {} || acc: {}\n'.format(e, index, acc[0], acc[1]))
                ##for x_test, y_test in zip(X_test, Y_test):
                ##    prediction = model.evaluate(np.array([x_test]),np.array([y_test]), batch_size=1)
                ##    print (prediction)
            index += 1
        acc_train /= len(X_train)
        sys.stderr.write('Epoch: {} || Train acc: {}\n'.format(e, acc_train))
        with open('train_acc_1000_mouse.log', 'a') as f:
            f.write('Epoch: {} || Train acc: {}\n'.format(e, acc_train))
        acc_test = 0
        for x_test, y_test in zip(X_test, Y_test):
            #acc_test = model.predict_classes(np.array([x_test]), batch_size=1)
            acc = model.evaluate(np.array([x_test]),np.array([y_test]), batch_size=1)
            acc_test += acc[1]
        acc_test /= len(X_test)
        test_history[e].append(acc_test)
        sys.stdout.write('Epoch: {} || Test acc: {}\n'.format(e, acc_test))
        with open('test_acc_1000_mouse.log', 'a') as f:
            f.write('Epoch: {} || Test acc: {}\n'.format(e, acc_test))

        model.save('lstm-dropout_025_recur_dropout_025-epoch-{}_1000_mouse.h5'.format(e))

    return model, train_history


# In[11]:


gene_cds = '../data/mm10/input/genes_cds.json'
gene_lengths = '../data/mm10/input/genes_lengths.json'


# In[22]:


genes_dict, gene_seq = load_data(gene_cds, gene_lengths, genes_to_keep=1000)


# In[29]:


training_genes, test_genes = split_train_test_genes(genes_dict, train_proportion=0.7)

## Shuffle the genes once again to avoid any bin wise correlation

np.random.shuffle(training_genes)
np.random.shuffle(test_genes)

X_train = []
Y_train = []

X_test = []
Y_test = []

for gene in training_genes:
    X, Y = one_hot_encoding(gene_seq[gene])
    X_train.append(X)
    Y_train.append(Y)

for gene in test_genes:
    X, Y = one_hot_encoding(gene_seq[gene])
    X_test.append(X)
    Y_test.append(Y)


# In[ ]:


model, trainhist = train(X_train, Y_train, X_test, Y_test)
model.save('lstm-dropout_025_recur_dropout_025-all_1000_mouse.h5')


# In[ ]:


with open('train_hist_025_recur_dropout_025_1000_mouse.pickle', 'wb') as f:
    pickle.dump(trainhist, f)

