#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: CRF_LSTM_BERT
======
配置文件：cmed.dl.crf_lstm_bert.norm.json
"""
import argparse
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.dl.crf_nn_bert import CRF_NN_BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class CRF_LSTM_BERT(CRF_NN_BERT):
    def __init__(self, **config):
        super().__init__(**config)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.layers,
                            bidirectional=self.bidirectional, batch_first=True)
        self.output_to_tag = nn.Linear(self.hidden_dim * self.n_directions, self.tags_size)

    def _init_hidden(self, batch_size):
        hidden = (torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.layers * self.n_directions, batch_size, self.hidden_dim).to(device))
        return hidden

    def forward(self, seq_list):
        batch = self.tokenizer(seq_list, padding=True, truncation=True, return_tensors="pt").to(device)
        embedded = self.get_token_embedding(batch, 0)
        embedded = self.del_special_token(seq_list, embedded)  # 剔除[CLS], [SEP]标识

        batch_size = len(seq_list)
        hidden = self._init_hidden(batch_size)
        output, _ = self.lstm(embedded, hidden)
        output = self.output_to_tag(output)
        output = output.reshape(-1, output.shape[2])
        tag_scores = F.log_softmax(output, dim=1)  # [batch_size*seq_len, tags_size]

        crf_score, seqs_tag = self.viterbi_decode(tag_scores)

        return tag_scores, crf_score, seqs_tag


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
