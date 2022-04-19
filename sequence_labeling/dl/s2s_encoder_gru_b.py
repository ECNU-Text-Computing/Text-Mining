#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Sequence to Sequence
======
A class for ...
"""

import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.data_processor import DataProcessor
from .s2s_encoder_gru import EncoderGRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class EncoderGRU_B(EncoderGRU):
    def __init__(self):
        super(EncoderGRU_B, self).__init__()
        config_path = './config/cmed/dl/cmed.dl.s2s_encoder_gru.norm.json'
        config = json.load(open(config_path, 'r'))
        # print(config)
        self.model_name = config['model_name']
        self.data_root = config['data_root']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.bidirectional = True if config['bidirectional'] == 'True' else False
        self.n_directions = 2 if self.bidirectional else 1  # 双向循环，输出的hidden是正向和反向hidden的拼接，所以要 *2
        self.dropout = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        self.num_layers = config['num_layers']
        self.output_size = self.hidden_dim

        vocab = DataProcessor(**config).load_vocab()
        self.vocab_size = len(vocab)
        tags = DataProcessor(**config).load_tags()
        self.tags_size = len(tags)
        self.SOS_token = tags['SOS']
        self.EOS_token = tags['EOS']
        self.index_tag_dict = dict(zip(tags.values(), tags.keys()))

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, bidirectional=self.bidirectional)  # GRU循环神经网络
        self.out = nn.Linear(self.hidden_dim * self.n_directions, self.output_size)  # 全连接层

        print('Initialized Model: {}.'.format(self.model_name))

    def init_hidden(self, batch_size):  # 初始化hidden
        return torch.zeros(self.num_layers * self.n_directions, batch_size, self.hidden_dim, device=device)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        embedded = self.embedding(input).view(-1, batch_size, self.embedding_dim)
        output, hidden = self.gru(embedded, hidden)
        output = F.log_softmax(self.out(output), dim=1)

        return output, hidden

if __name__ == '__main__':

    print(type(EncoderGRU_B()).__name__)