#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
RNNAttention
======
A class for something.
"""

import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from Deep.Base_Model import Base_Model
from utils.metrics import cal_all

class Attention(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, hidden_dim2,num_classes,num_layers,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)

        self.hidden_dim2=hidden_dim2
        self.num_layers=num_layers

        #2层双向 LSTM batch_size为第一维度
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim* 2))
        #nn.Parameter ：将固定的tensor转换成可训练的parameter 定义一个参数向量 query
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2))  #shape=hidden_dim * 2 #size=[256]
        self.tanh2 = nn.Tanh()
        #隐层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim2)
        #输出层
        self.fc = nn.Linear(hidden_dim2,num_classes)

    def forward(self, x):
        # TextAttention: input -> embedding -> LSTM -> KEY,QUERY,VALUE -> 隐层->->output

        # input x: [batch_size, seq_len] = [3, 5]
        #print(x.size())    # [batch_size, seq_len] = [ 3, 5 ]
        emb = self.embedding(x)  # [batch_size, seq_len, embed_dim]=[3, 5, 64]
        #print(emb.size())
        # num_direction:是否双向而取值为1或2，此处为是，
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_dim * num_direction]=[3, 5, 256]
        #print(H.size())
        #各时刻隐状态通过tanh激活函数后的输出作为KEY
        M = self.tanh1(H)  # [batch_size, seq_len, hidden_dim * num_direction] =[ 3, 5 ,256]
       # print(M.size())
        # M = torch.tanh(torch.matmul(H, self.u))
        #KEY 和 query 相乘，再通过softmax得到每个时刻对应的权重
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)   # [3,5, 1]
       # print(alpha.size())
        #各时刻的权重和各时刻的隐状态 Value 对应相乘
        out = H * alpha  # [3, 5, 256]
       # print(out.size())
        #再相加
        out = torch.sum(out, 1)  # [3,256]
       # print(out.size())
        out = F.relu(out)
        out = self.fc1(out) #  [batch_size,hidden2]
       # print(out.size())
        out = self.fc(out)  # [128, 64] [batch_size,classes]
       # print(out.size())
        return out
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    # python3 text_attention.py --phase test
    if args.phase == 'test':
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, hidden_dim2, num_classes, num_layers,\
        dropout_rate, learning_rate, num_epochs, batch_size,\
        criterion_name, optimizer_name, gpu=\
            200, 64,128, 64, 2, 2, 0.5, 0.001, 2, 128, 'CrossEntropyLoss', 'Adam', 1

        # new an objective.
        model = Attention( vocab_size, embed_dim, hidden_dim, hidden_dim2,num_classes,num_layers,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu)

        # a simple example of the input.
        x = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
        x = torch.LongTensor(x)  # input: [batch_size, seq_len] = [3, 5]

        # the designed model can produce an output.
        out = model(x)
        print(out)
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')
