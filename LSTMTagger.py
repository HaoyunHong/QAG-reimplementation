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

BATCH_SIZE = 1

class LSTMTagger(nn.Module):

    def __init__(self, vector_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.vector_dim=vector_dim
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(vector_dim, hidden_dim, bidirectional=True)

        # 然后应该把key words也训练了，然后把各个量cat起来，

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(in_features = hidden_dim, out_features = tagset_size)

    def forward(self, batch):
        pair1=batch[:,0].view(1, BATCH_SIZE, self.vector_dim)   # reshape x to  (seq_len, batch, input_size)
        pair2=batch[:,1].view(1, BATCH_SIZE, self.vector_dim)   # reshape x to  (seq_len, batch, input_size)
        _, (lstm_out1,_1) = self.lstm(pair1.float())
        _, (lstm_out2,_2)= self.lstm(pair2.float())
        fc_vec=lstm_out1-lstm_out2
        fc_vec=fc_vec[0]
        tag_space = self.hidden2tag(fc_vec)
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores
