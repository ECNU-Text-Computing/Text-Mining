#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
LSTM、BiLSTM
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

        self.num_layers = kwargs['num_layers']          #神经网络的层数，可以自己定义,在json里改
        self.num_directions = 2                         #取值为1或2，1为单向LSTM，2为双向LSTM，按照需求更改
        self.lstm = nn.LSTM(embed_dim, hidden_dim, self.num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.fc1 = nn.Linear(embed_dim * self.num_directions, hidden_dim * self.num_directions)
        self.fc_out = nn.Linear(hidden_dim * self.num_directions, num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, x):
        input = self.embedding(x)
        output, _ = self.lstm(input)

        out = self.dropout(output[:, -1, :])
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc_out(out)

        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done LSTM_Model!')


