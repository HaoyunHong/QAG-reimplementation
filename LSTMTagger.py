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

BATCH_SIZE = 4


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, input_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, dropout=0.2,
                            num_layers=2)  # 也许dropout还需要调

        # The linear layer that maps from hidden state space to tag space
        self.fc1 = nn.Linear(in_features=hidden_dim * 6 + 2 + 1, out_features=hidden_dim)

        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=tagset_size)

    def forward(self, batch):
        pair1 = batch[:, 0].view(1, BATCH_SIZE, self.embedding_dim)  # reshape x to  (seq_len, batch, input_size)
        pair2 = batch[:, 1].view(1, BATCH_SIZE, self.embedding_dim)
        keyword1 = batch[:, 2].view(1, BATCH_SIZE, self.embedding_dim)
        keyword2 = batch[:, 3].view(1, BATCH_SIZE, self.embedding_dim)
        jaccard = batch[:, 0, 0:2].view(1, BATCH_SIZE, 2)
        inverse_pairs = batch[:, 1, 0].view(1, BATCH_SIZE, 1)

        # pair, key分别过2层的lstm cell
        _, (ui, _ui) = self.lstm(pair1)
        _, (vi, _vi) = self.lstm(keyword1)
        _, (uj, _uj) = self.lstm(pair2)
        _, (vj, _vj) = self.lstm(keyword2)
        # output last timestep
        ui = ui[-1].view(1, BATCH_SIZE, self.hidden_dim)
        uj = uj[-1].view(1, BATCH_SIZE, self.hidden_dim)
        vi = vi[-1].view(1, BATCH_SIZE, self.hidden_dim)
        vj = vj[-1].view(1, BATCH_SIZE, self.hidden_dim)

        du = ui - uj
        dv = vi - vj

        # 连接起来的predictors放入全连接层
        fc_vec = torch.cat([ui[0], vi[0], uj[0], vj[0], du[0], dv[0], jaccard[0], inverse_pairs[0]], 1)  # 应该是要去掉一层括号的
        # 再过一个relu
        tag_space0 = F.relu(self.fc1(fc_vec))
        tag_space = self.fc2(tag_space0)
        # 然后变成probability
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores
