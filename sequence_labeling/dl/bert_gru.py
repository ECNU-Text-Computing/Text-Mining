#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: Bert_GRU
======
A class for GRU using Bert embedding.
配置文件：cmed.dl.bert_gru.norm.json
"""
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_labeling.dl.bert_mlp import Bert_MLP

torch.manual_seed(1)


class Bert_GRU(Bert_MLP):
    def __init__(self, **config):
        super().__init__(**config)

        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                          num_layers=self.layers, batch_first=True, bidirectional=self.bidirectional)  # GRU循环神经网络

        self.output_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tags_size)

    def _init_hidden(self, batch_size):
        return torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim)

    def forward(self, seq_list):
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt")
        embedded = self.bert_model(**batch).hidden_states[0]
        embedded = self.del_special_token(seq_list, embedded)  # 剔除[CLS], [SEP]标识

        batch_size = len(seq_list)
        hidden = self._init_hidden(batch_size)
        output, _ = self.gru(embedded, hidden)

        output = self.output_to_tag(output)
        output = output.reshape(-1, output.shape[2])
        tag_scores = F.log_softmax(output, dim=1)  # [batch_size*seq_len, tags_size]

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
