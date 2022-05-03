#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: BiLSTM
======
A class for BiLSTM.
配置文件：cmed.dl.bilstm.norm.json
"""
import argparse
import datetime
import sys

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.dl.base_model import BaseModel

torch.manual_seed(1)


class BiLSTM(BaseModel):
    def __init__(self, **config):
        super().__init__(**config)

    # torch.randn()随机初始化，随机数满足标准正态分布（0~1）/ torch.zeros()初始化参数为0
    def _init_hidden(self, batch_size):
        hidden = (torch.randn(self.layers * self.n_directions, batch_size, self.hidden_dim),
                  torch.randn(self.layers * self.n_directions, batch_size, self.hidden_dim))
        return hidden

    def forward(self, X, X_lengths, Y):
        batch_size, seq_len = X.size()
        hidden = self._init_hidden(batch_size)
        embeded = self.word_embeddings(X)
        embeded = rnn_utils.pack_padded_sequence(embeded, X_lengths, batch_first=True)
        output, _ = self.lstm(embeded, hidden)  # 使用初始化值
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        out = output.reshape(-1, output.shape[2])
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
