#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
LSTM、BiLSTM
======
A class for something.
======
LSTM是RNN的一种变体，它对RNN进行了改进并有效避免了常规RNN梯度爆炸或梯度消失的问题

LSTM的输入
    1、input(seq_len, batch, input_size)
    2、h0(num_layers * num_directions, batch, hidden_size)初始的隐藏状态
    3、c0(num_layers * num_directions, batch, hidden_size)初始的单元状态

LSTM的输出
    1、ontput(seq_len, batch, num_directions * hidden_size)保存了最后一层，每个time_step的输出,最后一项output[:, -1, :]
    2、hn(num_layers * num_directions, batch, hidden_size)最后时刻的隐藏状态
    3、cn(num_layers * num_directions, batch, hidden_size)最后时刻的单元状态，一般用不到
======
"""

import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.metrics import cal_all
from Deep.Base_Model import Base_Model

#LSTM、Bi-LSTM文本分类

class LSTM(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(LSTM, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)

        #self.num_layers = 2                             #测试文件时没法调用config文件，所以直接设置神经网络的曾是
        self.num_layers = kwargs['num_layers']          #神经网络的层数，可以自己取值,在json里改
        self.num_directions = 2                         #取值为1或2，1为单向LSTM，2为双向LSTM，按照需求更改

        self.lstm = nn.LSTM(embed_dim, hidden_dim, self.num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)

        self.fc1 = nn.Linear(embed_dim * self.num_directions, hidden_dim * self.num_directions)
        self.fc_out = nn.Linear(hidden_dim * self.num_directions, num_classes)
        #self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, x):
        input = self.embedding(x)
        output, _ = self.lstm(input)

        out = self.dropout(output[:, -1, :])  # 最后时刻的hidden state
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc_out(out)

        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    #python3 text_lstm.py --phase test
    if args.phase == 'test':
        print('This is a test process.')
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, num_classes,  \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 200, 64, 64, 2, 0.5, 0.0001, 3, 64, 'CrossEntropyLoss', 'Adam', 0

        # new an objective.
        model = LSTM(vocab_size, embed_dim, hidden_dim, num_classes,
                               dropout_rate, learning_rate, num_epochs, batch_size,
                               criterion_name, optimizer_name, gpu)
        # a simple example of the input.
        input = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
        input = torch.LongTensor(input)  # input: [batch_size, seq_len] = [3, 5]

        # the designed model can produce an output.
        output = model(input)
        print(output)
        print('This is a test process.')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done LSTM_Model!')


