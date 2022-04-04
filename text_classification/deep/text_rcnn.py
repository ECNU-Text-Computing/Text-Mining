import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from Deep.Base_Model import Base_Model
from sklearn.feature_extraction.text import TfidfVectorizer
from Data_Loader import Data_Loader


class TextRCNN(Base_Model):

    """配置参数"""
    def __init__(self,vocab_size, embed_dim,filter_sizes, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, num_layers,require_improvement,pad_size,gpu):
       super(TextRCNN, self).__init__(vocab_size, embed_dim,filter_sizes, hidden_dim, num_classes,
                                  dropout_rate, learning_rate, num_epochs, batch_size,
                                  criterion_name, optimizer_name, num_layers,require_improvement,pad_size,gpu)
       self.filter_sizes=filter_sizes
       self.num_layers=num_layers
       self.lstm = nn.LSTM(embed_dim,hidden_dim, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_rate)
       self.maxpool = nn.MaxPool1d(pad_size)
       self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):

        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, (hidden_state,cell_state)= self.lstm(embed)
        pooled = nn.MaxPool2d((out.size(1), 1))(out)
        pooled = pooled.squeeze(1)
        pred = self.fc(pooled)
        return pred

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
