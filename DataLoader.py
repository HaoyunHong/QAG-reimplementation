import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import json
import codecs
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

glove_path = 'F:/glove.6B/'


class DataLoader(object):

    def __init__(self, path):
        # load data
        train_data = json.load(codecs.open(path, 'r', 'utf-8'))
        tokenizer = pickle.load(open("tokenizer", "rb"))
        stop_list = []
        with codecs.open('stoplist.txt', 'r', 'utf-8') as f:
            for word in f.readlines():
                stop_list.append(word[:-1])

        # split text
        for i, pair in enumerate(train_data.copy()):
            seq1 = []
            seq2 = []
            seq1 = pair[1].split(' ')
            seq2 = pair[2].split(' ')
            train_data[i] = [pair[0], seq1, seq2]

        max_seq_len = 0

        # delete stop words
        for i, pair in enumerate(train_data):
            newseq1 = []
            newseq2 = []
            for word in pair[1]:
                if word not in stop_list:
                    newseq1.append(word)
            pair[1] = newseq1
            for word in pair[2]:
                if word not in stop_list:
                    newseq2.append(word)
            pair[2] = newseq2
            # get max sequece length
            max_seq_len = max(max_seq_len, max(len(pair[1]), len(pair[2])))

        labels = []
        title2vec1 = []
        title2vec2 = []
        keyword1 = []
        keyword2 = []
        length1 = []
        length2 = []
        jaccard = []
        inverse_pairs = []

        title2vec1 = tokenizer.texts_to_sequences([p[1] for p in train_data])
        title2vec2 = tokenizer.texts_to_sequences([p[2] for p in train_data])

        for i, pair in enumerate(train_data):
            _length1, _length2, _keyword1, _keyword2, _jaccard, _inverse_pairs = self.preprocess(title2vec1[i],
                                                                                                 title2vec2[i],
                                                                                                 max_seq_len)
            labels.append(pair[0])
            length1.append(_length1)
            length2.append(_length2)
            keyword1.append(_keyword1)
            keyword2.append(_keyword2)
            # jaccard.append([np.float32(_jaccard)] * (multiple * 2))
            # self.inverse_pairs.append([np.float32(_inverse_pairs)] * multiple)
            # title2vec1 = pad_sequences(title2vec1, maxlen=msl)
            # title2vec2 = pad_sequences(title2vec2, maxlen=msl)

        # padding vectors
        for i, pair in enumerate(title2vec1.copy()):
            title2vec1[i] = [0] * (max_seq_len - len(title2vec1.copy()[i])) + title2vec1.copy()[i]
        for i, pair in enumerate(title2vec2.copy()):
            title2vec2[i] = [0] * (max_seq_len - len(title2vec2.copy()[i])) + title2vec2.copy()[i]

        # text to vector with labels
        label_train_val = []
        for i in range(len(title2vec1)):
            label_train_val.append([labels[i], title2vec1[i], title2vec2[i]])

        # text to vector
        title2vec = []
        for i in range(len(title2vec1)):
            title2vec.append([title2vec1[i], title2vec2[i]])

        title2vec = torch.tensor(title2vec, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        self.title2vec = title2vec
        self.labels = labels
        self.max_seq_len = max_seq_len

    def compute_inverse_pairs(self, seq1, seq2, overlap, max_seq_len):
        look_up = {}
        new_seq1 = []
        new_seq2 = []
        for w in seq1:
            if w in overlap:
                look_up[w] = len(look_up) + 1
                new_seq1.append(look_up[w])
        for w in seq2:
            if w in overlap:
                new_seq2.append(look_up[w])
        result = 0
        for i in range(len(new_seq2)):
            for j in range(i, len(new_seq2)):
                if new_seq2[j] < i + 1:
                    result -= 1
        return result, \
               [0] * (max_seq_len - len(new_seq1)) + new_seq1 if len(new_seq1) <= max_seq_len else new_seq1[
                                                                                                   :max_seq_len], \
               [0] * (max_seq_len - len(new_seq2)) + new_seq2 if len(new_seq2) <= max_seq_len else new_seq2[
                                                                                                   :max_seq_len]

    def preprocess(self, seq1, seq2, max_seq_len):
        overlap = set(seq1).intersection(seq2)
        jaccard = len(overlap) / (len(seq1) + len(seq2) - len(overlap))
        inverse_pairs, keyword_seq1, keyword_seq2 = self.compute_inverse_pairs(seq1, seq2, overlap, max_seq_len)
        return len(seq1), len(seq2), keyword_seq1, keyword_seq2, jaccard, inverse_pairs
