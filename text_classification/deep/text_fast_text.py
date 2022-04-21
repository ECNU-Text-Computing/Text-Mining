#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
FastText
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

#FastText文本分类

class FastText(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(FastText, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)

        self.n_gram_vocab = kwargs['n_gram_vocab']  #词表大小，如250499
        self.embedding_ngram2 = nn.Embedding(self.n_gram_vocab, embed_dim)
        self.embedding_ngram3 = nn.Embedding(self.n_gram_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_dim * 3, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        out_word = self.embedding(x)   #生成用来表征文档的向量
        out_bigram = self.embedding_ngram2(x)   #二元有问题
        out_trigram = self.embedding_ngram3(x)  #三元有问题
        fasttext_out = torch.cat((out_word, out_bigram, out_trigram), -1)  #叠加所有词及n-gram的词向量
        #fasttext_out1 = torch.stack((out_word, out_bigram, out_trigram), 0)
        out = fasttext_out.mean(dim=1)  #跨列求平均
        #print('11',fasttext_out.shape,out.shape)

        out = self.dropout(out)
        #print("5555", out.size())
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

    print('Done FastText_Model!')
