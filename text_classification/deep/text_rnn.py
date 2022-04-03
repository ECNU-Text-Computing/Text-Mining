import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils.metrics import cal_all
from Deep.Base_Model import Base_Model

# 循环神经网络 (many-to-one)

class TextRNN(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name,num_layers, gpu,**kwargs):
        super(TextRNN, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, num_layers, gpu,**kwargs)


        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out = self.embedding(x) 
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
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

    print('Done Base_Model!')
