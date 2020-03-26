from LSTMTagger import LSTMTagger
from DataLoader import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.optim import lr_scheduler
import json
import codecs
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

from tensorboardX import SummaryWriter


# BiLSTM
data=DataLoader('train.txt')

EMBEDDING_DIM = data.max_seq_len
HIDDEN_DIM = data.max_seq_len
BATCH_SIZE = 1

torch.manual_seed(1)

# 准备训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(data.title2vec, data.labels, test_size=0.1)

sub_train_dataset = Data.TensorDataset(X_train, y_train)
sub_train_loader = Data.DataLoader(
    dataset=sub_train_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
    )

sub_valid_dataset = Data.TensorDataset(X_valid, y_valid)
sub_valid_loader = Data.DataLoader(
    dataset=sub_valid_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
    )

def adjust_learning_rate(optimizer, epoch, lr):
    lr *= (0.5 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(sub_train_loader), 2)

with SummaryWriter(comment='model')as w:

         w.add_graph(model, (torch.randn(1),))

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = model.parameters(), lr=0.01)

lr_init = optimizer.param_groups[0]['lr']

# strat training
for epoch in range(10000):  # again, normally you would NOT do 300 epochs, it is toy data
    train_accuracy=0
    train_loss=0
    valid_accuracy=0
    valid_loss=0

    # learning rate decay
    adjust_learning_rate(optimizer, epoch, lr_init)
    
    for step, (batch_x, batch_y) in enumerate(sub_train_loader):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        tag_scores = model(batch_x)

        prediction = torch.max(tag_scores, 1)[1] # 返回每行最大的数的索引
        pred_y = prediction.data.numpy()
        target_y = batch_y.data.numpy()
        train_accuracy += (pred_y == target_y).astype(int).sum()
        loss = loss_function(tag_scores, batch_y)
        train_loss+=loss

        if(epoch%10==0 and step==len(sub_train_loader)-1):
            f = open("training_history.txt", "a")
            print("epoch: {}".format(epoch)) 
            print("train loss = {:.4f}".format(train_loss/(step+1)))
            print("train accuracy = {:.2f}%".format(train_accuracy*100/(step+1)))
            print("epoch: {}".format(epoch), file=f) 
            print("train loss = {:.4f}".format(train_loss/(step+1)), file=f)
            print("train accuracy = {:.2f}%".format(train_accuracy*100/(step+1)), file=f)
            f.close()
                
        loss.backward()
        optimizer.step()
            

    for step0, (batch_x, batch_y) in enumerate(sub_valid_loader):
        with torch.no_grad():
            tag_scores = model(batch_x)
            valid_loss += loss_function(tag_scores, batch_y)
            prediction = torch.max(tag_scores, 1)[1] # 返回每行最大的数的索引
            pred_y = prediction.data.numpy()
            target_y = batch_y.data.numpy()
            valid_accuracy += (pred_y == target_y).astype(int).sum()
            if(epoch%10==0 and step0==len(sub_valid_loader)-1):
                f = open("training_history.txt", "a")
                print("valid loss = {:.4f}".format(valid_loss/(step0+1)))
                print("valid accuracy = {:.2f}%".format(valid_accuracy*100/(step0+1)))
                print("valid loss = {:.4f}".format(valid_loss/(step0+1)), file=f)
                print("valid accuracy = {:.2f}%".format(valid_accuracy*100/(step0+1)), file=f)
                f.close()
                
    # 每100个保存一次模型            
    if(epoch>0 and epoch%100==0):
            PATH ='model_biLSTM_epoch_' + str(epoch) +'.ckpt'
            torch.save(model.state_dict(), PATH)



