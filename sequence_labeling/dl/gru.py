#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: GRU
======
A class for GRU.
"""

import argparse
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.dl.base_model import BaseModel

torch.manual_seed(1)


class GRU(BaseModel):
    def __init__(self, **config):
        super().__init__(**config)

        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                          num_layers=self.layers, batch_first=True, bidirectional=self.bidirectional,
                          dropout=self.dropout_p)  # GRU循环神经网络

    def _init_hidden(self, batch_size):
        return torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim)

    def forward(self, X, X_lengths):
        batch_size, seq_len = X.size()
        hidden = self._init_hidden(batch_size)
        embeded = self.word_embeddings(X)
        embeded = rnn_utils.pack_padded_sequence(embeded, X_lengths, batch_first=True)
        output, _ = self.gru(embeded, hidden)  # 使用初始化值
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        out = output.contiguous()
        out = out.view(-1, out.shape[2])
        out = self.output_to_tag(out)

        tag_scores = F.log_softmax(out, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores


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
