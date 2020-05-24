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
import re
import bcolz

glove_path = 'F:/glove.6B/'
embed_dim = 50
padding = 0  # 对应的是填空符


class DataLoader(object):

    def __init__(self, path):
        # load data
        train_data = json.load(codecs.open(path, 'r', 'utf-8'))
        stop_list = []

        with codecs.open('stoplist.txt', 'r', 'utf-8') as f:
            for word in f.readlines():
                stop_list.append(word[:-1])

        # split text
        for i, pair in enumerate(train_data.copy()):
            seq1 = re.split('[ —:]', pair[1])  # 空格和连接符都为分隔符，并且用[]作分割
            seq2 = re.split('[ —:]', pair[2])
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
            # get max sequence length
            max_seq_len = max(max_seq_len, max(len(pair[1]), len(pair[2])))

        # 去除重复的词
        word_to_id = {}
        id_to_word = {}
        # 找到字典里的词
        count = 0
        for i, pair in enumerate(train_data):
            for j, _ in enumerate(pair[1]):
                try:
                    _ = word_to_id[pair[1][j]]
                except KeyError:
                    count = count + 1
                    word_to_id[pair[1][j]] = count
                    id_to_word[count] = pair[1][j]
            for k, _ in enumerate(pair[2]):
                try:
                    _ = word_to_id[pair[2][k]]
                except KeyError:
                    count = count + 1
                    word_to_id[pair[2][k]] = count
                    id_to_word[count] = pair[2][k]

        target_vocab = ['*']
        for i in range(count):
            target_vocab.append(id_to_word[i + 1])

        vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
        words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
        # word embedding 字典
        glove = {w: vectors[word2idx[w]] for w in words}

        matrix_len = len(target_vocab)
        weights_matrix = np.zeros((matrix_len, 50))
        words_found = 0

        for i, word in enumerate(target_vocab[1:]):
            try:
                weights_matrix[i] = glove[word]
                words_found += 1
            # 不在预训练模型里的词就给个最初的id
            except KeyError:
                weights_matrix[i] = word_to_id[word]

        labels = []
        title2vec1 = []
        title2vec2 = []
        keyword1 = []
        keyword2 = []
        length1 = []
        length2 = []
        jaccard = []
        inverse_pairs = []
        sentence_emb1 = []
        sentence_emb2 = []

        # tokenizer
        for i, p in enumerate(train_data):
            title2vec1.append([])
            for w in p[1]:
                title2vec1[i].append(word_to_id[w])

        for i, p in enumerate(train_data):
            title2vec2.append([])
            for w in p[2]:
                title2vec2[i].append(word_to_id[w])

        # sentence embedding mean
        for i, p in enumerate(title2vec1):
            sentence_emb1.append([0]*embed_dim)
            for wid in p:
                for j in range(0, embed_dim):
                    sentence_emb1[i][j] += weights_matrix[wid][j]
            for k in range(0, embed_dim):
                sentence_emb1[i][k] = sentence_emb1[i][k]/len(p)

        for i, p in enumerate(title2vec2):
            sentence_emb2.append([0]*embed_dim)
            for wid in p:
                for j in range(0, embed_dim):
                    sentence_emb2[i][j] += weights_matrix[wid][j]
            for k in range(0, embed_dim):
                sentence_emb2[i][k] = sentence_emb2[i][k]/len(p)

        for i, pair in enumerate(train_data):
            _length1, _length2, _keyword1, _keyword2, _jaccard, _inverse_pairs = self.preprocess(title2vec1[i],
                                                                                                 title2vec2[i])

            # length1.append(_length1)
            # length2.append(_length2)
            keyword1.append(_keyword1)
            keyword2.append(_keyword2)
            # padding 让模型传参更方便
            labels.append([pair[0]] + [padding] * (embed_dim - 1))
            jaccard.append([np.float32(_jaccard)] * 2 + [padding] * (embed_dim - 2))
            inverse_pairs.append([np.float32(_inverse_pairs)] + [padding] * (embed_dim - 1))

        # all the predictors
        predictors = []
        for i, _ in enumerate(sentence_emb1):
            predictors.append([sentence_emb1[i], sentence_emb2[i], keyword1[i], keyword2[i], jaccard[i], inverse_pairs[i]])

        labels_hidden = []
        for i in range(len(sentence_emb1)):
            labels_hidden.append([labels[i]])

        predictors = torch.tensor(predictors, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        self.max_seq_len = max_seq_len
        self.predictors = predictors
        self.labels = labels
        self.embed_dim = embed_dim

    def compute_inverse_pairs(self, seq1, seq2, overlap):
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
               [0] * (embed_dim - len(new_seq1)) + new_seq1, \
               [0] * (embed_dim - len(new_seq2)) + new_seq2

    def preprocess(self, seq1, seq2):
        overlap = set(seq1).intersection(seq2)
        jaccard = len(overlap) / (len(seq1) + len(seq2) - len(overlap))
        inverse_pairs, keyword_seq1, keyword_seq2 = self.compute_inverse_pairs(seq1, seq2, overlap)
        return len(seq1), len(seq2), keyword_seq1, keyword_seq2, jaccard, inverse_pairs
